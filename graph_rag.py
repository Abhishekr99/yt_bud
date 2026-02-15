import hashlib
import json
import os
import re
from typing import Any, Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_core.prompts import ChatPromptTemplate

from utility import create_chunks, llm

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

CONCEPT_INDEX = "conceptIndex"
CHUNK_INDEX = "chunkTextIndex"


def graph_enabled() -> bool:
    return bool(NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD)


@st.cache_resource(show_spinner=False)
def get_neo4j_driver():
    if not graph_enabled():
        raise ValueError("Neo4j credentials are not configured.")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def _clean_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9]*", "", stripped, count=1).strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1:
        return stripped[start : end + 1]
    return stripped


def _normalize_concept(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    return key


def _hash_transcript(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def setup_graph_schema():
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(
            "CREATE CONSTRAINT video_id IF NOT EXISTS "
            "FOR (v:Video) REQUIRE v.video_id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT concept_key IF NOT EXISTS "
            "FOR (c:Concept) REQUIRE c.key IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        )
        session.run(
            "CREATE FULLTEXT INDEX "
            f"{CONCEPT_INDEX} IF NOT EXISTS FOR (c:Concept) ON EACH [c.name, c.key]"
        )
        session.run(
            "CREATE FULLTEXT INDEX "
            f"{CHUNK_INDEX} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]"
        )


def get_video_graph_state(video_id: str) -> Dict[str, Any]:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        record = session.run(
            "MATCH (v:Video {video_id: $video_id}) "
            "RETURN v.graph_ready AS graph_ready, v.transcript_hash AS transcript_hash",
            video_id=video_id,
        ).single()
    if not record:
        return {"exists": False, "graph_ready": False, "transcript_hash": None}
    return {
        "exists": True,
        "graph_ready": bool(record.get("graph_ready")),
        "transcript_hash": record.get("transcript_hash"),
    }


def clear_video_graph(video_id: str) -> None:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(
            "MATCH (v:Video {video_id: $video_id})-[:HAS_CHUNK]->(ch:Chunk) "
            "DETACH DELETE ch",
            video_id=video_id,
        )
        session.run(
            "MATCH (v:Video {video_id: $video_id}) DETACH DELETE v",
            video_id=video_id,
        )
        session.run(
            "MATCH (c:Concept) "
            "WHERE NOT (c)<-[:MENTIONS]-(:Chunk) "
            "DETACH DELETE c"
        )


def _extract_graph_entities(chunk_text: str, max_concepts: int = 10) -> Tuple[List[str], List[Dict[str, str]]]:
    prompt = ChatPromptTemplate.from_template(
        """
        Extract a compact knowledge graph from the transcript chunk.
        Return STRICT JSON only:
        {{
          "concepts": ["term1", "term2"],
          "relations": [{{"source": "term1", "relation": "related_to", "target": "term2"}}]
        }}
        Rules:
        - concepts: 4-{max_concepts} items, prefer nouns/proper nouns/acronyms.
        - relations: include only if explicitly stated in the chunk.
        - use short relation labels, like "prerequisite_of", "part_of", "causes".
        - do not invent concepts not present in the chunk.
        Chunk:
        {chunk}
        """
    )
    response = (prompt | llm).invoke(
        {"chunk": chunk_text, "max_concepts": max_concepts}
    )
    cleaned = _clean_json_text(response.content)
    try:
        payload = json.loads(cleaned)
        concepts = payload.get("concepts", [])
        relations = payload.get("relations", [])
    except json.JSONDecodeError:
        concepts = []
        relations = []

    if not concepts:
        fallback = re.findall(r"\b[A-Z][a-zA-Z0-9-]{2,}\b", chunk_text)
        concepts = list(dict.fromkeys(fallback))[:max_concepts]

    return concepts[:max_concepts], relations[: max_concepts * 2]


def ensure_video_graph(
    video_id: str,
    transcript: str,
    language: str,
    rebuild: bool = False,
) -> Dict[str, Any]:
    setup_graph_schema()
    transcript_hash = _hash_transcript(transcript)
    current_state = get_video_graph_state(video_id)
    if (
        current_state["exists"]
        and current_state["graph_ready"]
        and current_state["transcript_hash"] == transcript_hash
        and not rebuild
    ):
        return {"ready": True, "skipped": True, "reason": "already_built"}

    if current_state["exists"] and not rebuild:
        clear_video_graph(video_id)

    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(
            "MERGE (v:Video {video_id: $video_id}) "
            "SET v.language = $language, v.transcript_hash = $transcript_hash, "
            "v.graph_ready = false, v.updated_at = timestamp()",
            video_id=video_id,
            language=language,
            transcript_hash=transcript_hash,
        )

    docs = create_chunks(transcript)
    concept_count = 0
    relation_count = 0

    with driver.session(database=NEO4J_DATABASE) as session:
        for idx, doc in enumerate(docs):
            chunk_text = doc.page_content.strip()
            if not chunk_text:
                continue
            chunk_id = f"{video_id}::{idx}"
            session.run(
                "MERGE (v:Video {video_id: $video_id}) "
                "MERGE (ch:Chunk {chunk_id: $chunk_id}) "
                "SET ch.text = $text, ch.index = $index, ch.video_id = $video_id "
                "MERGE (v)-[:HAS_CHUNK]->(ch)",
                video_id=video_id,
                chunk_id=chunk_id,
                text=chunk_text,
                index=idx,
            )

            concepts, relations = _extract_graph_entities(chunk_text)
            normalized = []
            for concept in concepts:
                key = _normalize_concept(concept)
                if not key:
                    continue
                normalized.append((key, concept.strip()))

            for key, name in normalized:
                session.run(
                    "MERGE (c:Concept {key: $key}) "
                    "SET c.name = $name",
                    key=key,
                    name=name,
                )
                session.run(
                    "MATCH (ch:Chunk {chunk_id: $chunk_id}) "
                    "MATCH (c:Concept {key: $key}) "
                    "MERGE (ch)-[:MENTIONS {video_id: $video_id}]->(c)",
                    chunk_id=chunk_id,
                    key=key,
                    video_id=video_id,
                )
                concept_count += 1

            for relation in relations:
                source = _normalize_concept(relation.get("source", ""))
                target = _normalize_concept(relation.get("target", ""))
                label = relation.get("relation", "related_to").strip() or "related_to"
                if not source or not target:
                    continue
                session.run(
                    "MERGE (a:Concept {key: $source}) "
                    "MERGE (b:Concept {key: $target}) "
                    "MERGE (a)-[r:RELATED_TO {video_id: $video_id, relation: $label}]->(b)",
                    source=source,
                    target=target,
                    label=label,
                    video_id=video_id,
                )
                relation_count += 1

            keys = [key for key, _ in normalized]
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    session.run(
                        "MATCH (a:Concept {key: $a}) "
                        "MATCH (b:Concept {key: $b}) "
                        "MERGE (a)-[r:CO_OCCURS {video_id: $video_id}]->(b) "
                        "ON CREATE SET r.weight = 1 "
                        "ON MATCH SET r.weight = r.weight + 1",
                        a=keys[i],
                        b=keys[j],
                        video_id=video_id,
                    )

        session.run(
            "MATCH (v:Video {video_id: $video_id}) "
            "SET v.graph_ready = true, v.updated_at = timestamp()",
            video_id=video_id,
        )

    return {
        "ready": True,
        "skipped": False,
        "chunks": len(docs),
        "concept_mentions": concept_count,
        "relations": relation_count,
    }


def _format_history(chat_history: List[Dict[str, str]], history_turns: int) -> str:
    turns_to_keep = max(history_turns, 0) * 2
    recent_messages = chat_history[-turns_to_keep:] if turns_to_keep else []
    formatted = []
    for msg in recent_messages:
        role = msg.get("role", "").lower()
        if role not in ("user", "assistant"):
            continue
        speaker = "User" if role == "user" else "Assistant"
        formatted.append(f"{speaker}: {msg.get('content', '').strip()}")
    return "\n".join(formatted).strip() or "No prior chat turns."


def _query_concepts(session, video_id: str, query: str, limit: int):
    return session.run(
        "CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score "
        "WITH node, score "
        "MATCH (node)<-[:MENTIONS {video_id: $video_id}]-(ch:Chunk) "
        "RETURN node, score, collect(DISTINCT ch) AS chunks "
        "ORDER BY score DESC "
        "LIMIT $limit",
        index_name=CONCEPT_INDEX,
        query=query,
        video_id=video_id,
        limit=limit,
    )


def _query_chunks(session, video_id: str, query: str, limit: int):
    return session.run(
        "CALL db.index.fulltext.queryNodes($index_name, $query) YIELD node, score "
        "WHERE node.video_id = $video_id "
        "RETURN node, score "
        "ORDER BY score DESC "
        "LIMIT $limit",
        index_name=CHUNK_INDEX,
        query=query,
        video_id=video_id,
        limit=limit,
    )


def graph_rag_retrieve(video_id: str, question: str, max_chunks: int = 6) -> Tuple[str, Dict[str, Any]]:
    driver = get_neo4j_driver()
    concepts: Dict[str, str] = {}
    chunks: Dict[str, str] = {}
    relations: List[str] = []
    co_occurs: List[str] = []

    with driver.session(database=NEO4J_DATABASE) as session:
        concept_records = list(_query_concepts(session, video_id, question, limit=8))
        for record in concept_records:
            node = record["node"]
            if node:
                key = node.get("key")
                name = node.get("name") or node.get("key")
                if key:
                    concepts[key] = name
            for ch in record.get("chunks", []):
                chunk_id = ch.get("chunk_id")
                if chunk_id and chunk_id not in chunks:
                    chunks[chunk_id] = ch.get("text", "")

        if not chunks:
            chunk_records = list(_query_chunks(session, video_id, question, limit=max_chunks))
            for record in chunk_records:
                node = record["node"]
                if not node:
                    continue
                chunk_id = node.get("chunk_id")
                if chunk_id and chunk_id not in chunks:
                    chunks[chunk_id] = node.get("text", "")
                concept_records = session.run(
                    "MATCH (ch:Chunk {chunk_id: $chunk_id})-[:MENTIONS {video_id: $video_id}]->(c:Concept) "
                    "RETURN c",
                    chunk_id=chunk_id,
                    video_id=video_id,
                )
                for concept in concept_records:
                    cnode = concept.get("c")
                    if cnode:
                        key = cnode.get("key")
                        name = cnode.get("name") or key
                        if key:
                            concepts[key] = name

        if concepts:
            keys = list(concepts.keys())
            rel_records = session.run(
                "MATCH (a:Concept)-[r:RELATED_TO {video_id: $video_id}]->(b:Concept) "
                "WHERE a.key IN $keys AND b.key IN $keys "
                "RETURN a.name AS source, r.relation AS relation, b.name AS target "
                "LIMIT 12",
                video_id=video_id,
                keys=keys,
            )
            for record in rel_records:
                relations.append(
                    f"{record['source']} -{record['relation']}- {record['target']}"
                )

            co_records = session.run(
                "MATCH (a:Concept)-[r:CO_OCCURS {video_id: $video_id}]->(b:Concept) "
                "WHERE a.key IN $keys AND b.key IN $keys "
                "RETURN a.name AS source, b.name AS target, r.weight AS weight "
                "ORDER BY r.weight DESC "
                "LIMIT 10",
                video_id=video_id,
                keys=keys,
            )
            for record in co_records:
                co_occurs.append(
                    f"{record['source']} <-> {record['target']} (weight {record['weight']})"
                )

    concept_list = "\n".join(f"- {name}" for name in concepts.values())
    relation_list = "\n".join(f"- {item}" for item in relations)
    co_list = "\n".join(f"- {item}" for item in co_occurs)
    chunk_list = "\n\n".join(
        f"[Chunk {idx + 1}]\n{text}"
        for idx, text in enumerate(list(chunks.values())[:max_chunks])
        if text
    )

    context_sections = []
    if concept_list:
        context_sections.append(f"Concepts:\n{concept_list}")
    if relation_list:
        context_sections.append(f"Relations:\n{relation_list}")
    if co_list:
        context_sections.append(f"Co-occurrence:\n{co_list}")
    if chunk_list:
        context_sections.append(f"Transcript Chunks:\n{chunk_list}")

    context = "\n\n".join(context_sections).strip()
    stats = {
        "concepts": len(concepts),
        "chunks": min(len(chunks), max_chunks),
        "relations": len(relations),
        "co_occurrences": len(co_occurs),
        "context_chars": len(context),
    }
    return context, stats


def graph_rag_answer(
    question: str,
    video_id: str,
    chat_history: List[Dict[str, str]] = None,
    history_turns: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    chat_history = chat_history or []
    history_text = _format_history(chat_history, history_turns)
    context, stats = graph_rag_retrieve(video_id, question)
    if not context:
        return (
            "I couldn't find that information in the transcript. "
            "Could you rephrase or ask something else?",
            stats,
        )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a precise assistant answering questions about a video using graph context.
        - Use the graph context and transcript chunks to answer.
        - Do NOT add greetings; reply directly to the user's request.
        - Answer ONLY using the provided context/history; avoid outside knowledge.
        - If the answer is not in the context, reply:
          "I couldn't find that information in the transcript. Could you rephrase or ask something else?"
        - Keep responses concise and easy to skim.

        Recent Chat History:
        {history}

        Graph Context:
        {context}

        User Question:
        {question}

        Answer:
        """
    )

    response = (prompt | llm).invoke(
        {"context": context, "question": question, "history": history_text}
    )
    return response.content, stats


def get_video_concepts(video_id: str) -> List[str]:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        records = list(
            session.run(
            "MATCH (ch:Chunk {video_id: $video_id})-[:MENTIONS {video_id: $video_id}]->(c:Concept) "
            "RETURN DISTINCT c.key AS key, c.name AS name",
            video_id=video_id,
            )
        )
    concepts = []
    for record in records:
        name = record.get("name") or record.get("key")
        if name:
            concepts.append(name)
    return sorted(set(concepts))


def compare_video_concepts(video_id_a: str, video_id_b: str) -> Dict[str, List[str]]:
    concepts_a = set(get_video_concepts(video_id_a))
    concepts_b = set(get_video_concepts(video_id_b))
    return {
        "only_in_a": sorted(concepts_a - concepts_b),
        "only_in_b": sorted(concepts_b - concepts_a),
        "shared": sorted(concepts_a & concepts_b),
    }


def compare_with_reference(video_id: str, reference_terms: List[str]) -> Dict[str, List[str]]:
    concepts = set(get_video_concepts(video_id))
    normalized_ref = {term.strip() for term in reference_terms if term.strip()}
    return {
        "missing": sorted(normalized_ref - concepts),
        "covered": sorted(concepts & normalized_ref),
    }
