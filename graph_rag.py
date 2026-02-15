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
ALLOWED_RELATIONS = {
    "prerequisite_of",
    "explains",
    "depends_on",
    "part_of",
    "used_for",
    "contrasts_with",
    "example_of",
    "defines",
    "related_to",
}


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


def _extract_graph_entities(
    chunk_text: str, max_concepts: int = 10
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt = ChatPromptTemplate.from_template(
        """
        Extract a compact knowledge graph from the transcript chunk.
        Return STRICT JSON only:
        {{
          "concepts": [{{"name": "term1", "aliases": ["alt"], "salience": 0.5}}],
          "relations": [{{"source": "term1", "relation": "prerequisite_of", "target": "term2", "confidence": 0.6, "evidence": "short phrase"}}]
        }}
        Rules:
        - concepts: 4-{max_concepts} items, prefer nouns/proper nouns/acronyms.
        - relations: include only if explicitly stated in the chunk.
        - use relation labels from: prerequisite_of, explains, depends_on, part_of, used_for, contrasts_with, example_of, defines.
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
        concepts_raw = payload.get("concepts", [])
        relations_raw = payload.get("relations", [])
    except json.JSONDecodeError:
        concepts_raw = []
        relations_raw = []

    concepts: List[Dict[str, Any]] = []
    if isinstance(concepts_raw, list):
        for item in concepts_raw:
            if isinstance(item, str):
                name = item.strip()
                aliases: List[str] = []
                salience = 0.5
            elif isinstance(item, dict):
                name = (
                    item.get("name")
                    or item.get("concept")
                    or item.get("term")
                    or ""
                ).strip()
                aliases = item.get("aliases") or []
                if isinstance(aliases, str):
                    aliases = [aliases]
                salience = item.get("salience")
                try:
                    salience = float(salience)
                except (TypeError, ValueError):
                    salience = 0.5
            else:
                continue
            if name:
                concepts.append(
                    {"name": name, "aliases": aliases, "salience": salience}
                )

    relations: List[Dict[str, Any]] = []
    if isinstance(relations_raw, list):
        for item in relations_raw:
            if not isinstance(item, dict):
                continue
            source = (item.get("source") or item.get("from") or "").strip()
            target = (item.get("target") or item.get("to") or "").strip()
            relation = (item.get("relation") or item.get("type") or "related_to").strip()
            relation = relation.lower()
            if relation not in ALLOWED_RELATIONS:
                relation = "related_to"
            confidence = item.get("confidence")
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.5
            evidence = (item.get("evidence") or item.get("snippet") or "").strip()
            if source and target:
                relations.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": relation,
                        "confidence": confidence,
                        "evidence": evidence,
                    }
                )

    if not concepts:
        fallback = re.findall(r"\b[A-Z][a-zA-Z0-9-]{2,}\b", chunk_text)
        concepts = [
            {"name": name, "aliases": [], "salience": 0.4}
            for name in list(dict.fromkeys(fallback))[:max_concepts]
        ]

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
        prev_chunk_id = None
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
            if prev_chunk_id:
                session.run(
                    "MATCH (a:Chunk {chunk_id: $prev}) "
                    "MATCH (b:Chunk {chunk_id: $cur}) "
                    "MERGE (a)-[:NEXT {video_id: $video_id}]->(b)",
                    prev=prev_chunk_id,
                    cur=chunk_id,
                    video_id=video_id,
                )
            prev_chunk_id = chunk_id

            concepts, relations = _extract_graph_entities(chunk_text)
            normalized = []
            for concept in concepts:
                name = concept.get("name", "").strip()
                key = _normalize_concept(name)
                if not key:
                    continue
                normalized.append((key, name))

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
                    "MERGE (ch)-[m:MENTIONS {video_id: $video_id}]->(c) "
                    "SET m.chunk_index = $index",
                    chunk_id=chunk_id,
                    key=key,
                    video_id=video_id,
                    index=idx,
                )
                concept_count += 1

            for relation in relations:
                source = _normalize_concept(relation.get("source", ""))
                target = _normalize_concept(relation.get("target", ""))
                label = relation.get("relation", "related_to").strip().lower()
                if label not in ALLOWED_RELATIONS:
                    label = "related_to"
                evidence = relation.get("evidence", "").strip()
                confidence = relation.get("confidence", 0.5)
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.5
                if not source or not target:
                    continue
                session.run(
                    "MERGE (a:Concept {key: $source}) "
                    "MERGE (b:Concept {key: $target}) "
                    "MERGE (a)-[r:RELATED_TO {video_id: $video_id, relation: $label}]->(b) "
                    "SET r.evidence = coalesce(r.evidence, $evidence), "
                    "r.confidence = coalesce(r.confidence, $confidence)",
                    source=source,
                    target=target,
                    label=label,
                    video_id=video_id,
                    evidence=evidence,
                    confidence=confidence,
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


def _query_concepts(session, video_id: str, search_query: str, limit: int):
    return session.run(
        "CALL db.index.fulltext.queryNodes($index_name, $search_query) YIELD node, score "
        "WITH node, score "
        "MATCH (node)<-[:MENTIONS {video_id: $video_id}]-(ch:Chunk) "
        "RETURN node, score, collect(DISTINCT ch) AS chunks "
        "ORDER BY score DESC "
        "LIMIT $limit",
        index_name=CONCEPT_INDEX,
        search_query=search_query,
        video_id=video_id,
        limit=limit,
    )


def _query_chunks(session, video_id: str, search_query: str, limit: int):
    return session.run(
        "CALL db.index.fulltext.queryNodes($index_name, $search_query) YIELD node, score "
        "WHERE node.video_id = $video_id "
        "RETURN node, score "
        "ORDER BY score DESC "
        "LIMIT $limit",
        index_name=CHUNK_INDEX,
        search_query=search_query,
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


def get_video_concept_keys(video_id: str) -> Dict[str, str]:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        records = list(
            session.run(
                "MATCH (ch:Chunk {video_id: $video_id})-[:MENTIONS {video_id: $video_id}]->(c:Concept) "
                "RETURN DISTINCT c.key AS key, c.name AS name",
                video_id=video_id,
            )
        )
    return {
        record["key"]: (record.get("name") or record.get("key") or record["key"])
        for record in records
        if record.get("key")
    }


def get_first_occurrence_map(video_id: str) -> Dict[str, int]:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        records = list(
            session.run(
                "MATCH (ch:Chunk {video_id: $video_id})-[m:MENTIONS {video_id: $video_id}]->(c:Concept) "
                "RETURN c.key AS key, min(coalesce(m.chunk_index, ch.index)) AS first_idx",
                video_id=video_id,
            )
        )
    return {
        record["key"]: int(record["first_idx"])
        for record in records
        if record.get("key") is not None and record.get("first_idx") is not None
    }


def get_first_occurrence(video_id: str, concept_key: str) -> int:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        record = session.run(
            "MATCH (ch:Chunk {video_id: $video_id})-[m:MENTIONS {video_id: $video_id}]->(c:Concept {key: $key}) "
            "RETURN min(coalesce(m.chunk_index, ch.index)) AS first_idx",
            video_id=video_id,
            key=concept_key,
        ).single()
    if not record or record.get("first_idx") is None:
        return -1
    return int(record["first_idx"])


def get_concept_occurrences(
    video_id: str, concept_key: str, limit: int = 5, snippet_len: int = 220
) -> List[Dict[str, Any]]:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        records = list(
            session.run(
                "MATCH (ch:Chunk {video_id: $video_id})-[m:MENTIONS {video_id: $video_id}]->(c:Concept {key: $key}) "
                "RETURN coalesce(m.chunk_index, ch.index) AS idx, substring(ch.text, 0, $snippet_len) AS snippet "
                "ORDER BY idx ASC "
                "LIMIT $limit",
                video_id=video_id,
                key=concept_key,
                snippet_len=snippet_len,
                limit=limit,
            )
        )
    return [
        {"chunk_index": int(record["idx"]), "snippet": record.get("snippet", "")}
        for record in records
        if record.get("idx") is not None
    ]


def _get_concept_frequency(video_id: str, concept_key: str) -> Tuple[int, int]:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        record = session.run(
            "MATCH (ch:Chunk {video_id: $video_id})-[m:MENTIONS {video_id: $video_id}]->(c:Concept {key: $key}) "
            "RETURN count(m) AS mentions, min(coalesce(m.chunk_index, ch.index)) AS first_idx",
            video_id=video_id,
            key=concept_key,
        ).single()
    if not record:
        return 0, -1
    mentions = int(record.get("mentions", 0) or 0)
    first_idx = record.get("first_idx")
    return mentions, int(first_idx) if first_idx is not None else -1


def _get_concept_co_strength(video_id: str, concept_key: str) -> float:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        record = session.run(
            "MATCH (c:Concept {key: $key})-[r:CO_OCCURS {video_id: $video_id}]-(n:Concept) "
            "RETURN sum(r.weight) AS strength",
            video_id=video_id,
            key=concept_key,
        ).single()
    if not record or record.get("strength") is None:
        return 0.0
    return float(record["strength"])


def get_relations(video_id: str) -> List[Dict[str, Any]]:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        records = list(
            session.run(
                "MATCH (a:Concept)-[r:RELATED_TO {video_id: $video_id}]->(b:Concept) "
                "RETURN a.key AS skey, a.name AS sname, r.relation AS rel, "
                "b.key AS tkey, b.name AS tname, r.evidence AS evidence, r.confidence AS confidence",
                video_id=video_id,
            )
        )
    relations = []
    for record in records:
        if not record.get("skey") or not record.get("tkey"):
            continue
        relations.append(
            {
                "skey": record["skey"],
                "sname": record.get("sname") or record["skey"],
                "rel": record.get("rel") or "related_to",
                "tkey": record["tkey"],
                "tname": record.get("tname") or record["tkey"],
                "evidence": record.get("evidence") or "",
                "confidence": float(record.get("confidence") or 0.5),
            }
        )
    return relations


def get_window_concepts(
    video_id: str, start_idx: int, end_idx: int
) -> Dict[str, str]:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        records = list(
            session.run(
                "MATCH (v:Video {video_id: $video_id})-[:HAS_CHUNK]->(ch:Chunk) "
                "WHERE ch.index >= $start AND ch.index <= $end "
                "MATCH (ch)-[:MENTIONS {video_id: $video_id}]->(c:Concept) "
                "RETURN DISTINCT c.key AS key, c.name AS name",
                video_id=video_id,
                start=start_idx,
                end=end_idx,
            )
        )
    return {
        record["key"]: (record.get("name") or record.get("key") or record["key"])
        for record in records
        if record.get("key")
    }


def _get_max_chunk_index(video_id: str) -> int:
    driver = get_neo4j_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        record = session.run(
            "MATCH (ch:Chunk {video_id: $video_id}) RETURN max(ch.index) AS max_idx",
            video_id=video_id,
        ).single()
    if not record or record.get("max_idx") is None:
        return -1
    return int(record["max_idx"])


def _infer_prereqs_from_chunk(
    concept: str, snippet: str, max_items: int = 5
) -> List[str]:
    if not snippet:
        return []
    prompt = ChatPromptTemplate.from_template(
        """
        You are identifying prerequisite concepts implied by a transcript snippet.
        Return STRICT JSON only:
        {{"prerequisites": ["item1", "item2"]}}
        Rules:
        - List 1-{max_items} prerequisites only if clearly implied.
        - Keep each item short (1-3 words).
        - Do not invent unrelated items.

        Concept: {concept}
        Snippet: {snippet}
        """
    )
    response = (prompt | llm).invoke(
        {"concept": concept, "snippet": snippet, "max_items": max_items}
    )
    cleaned = _clean_json_text(response.content)
    try:
        payload = json.loads(cleaned)
        items = payload.get("prerequisites", [])
    except json.JSONDecodeError:
        items = re.findall(r"\"([^\"]+)\"", cleaned)
    return [item.strip() for item in items if isinstance(item, str) and item.strip()]


def _get_co_neighbors(video_id: str, concept_keys: List[str], limit: int = 5) -> Dict[str, float]:
    driver = get_neo4j_driver()
    weights: Dict[str, float] = {}
    with driver.session(database=NEO4J_DATABASE) as session:
        for key in concept_keys:
            records = list(
                session.run(
                    "MATCH (c:Concept {key: $key})-[r:CO_OCCURS {video_id: $video_id}]-(n:Concept) "
                    "RETURN n.key AS nkey, n.name AS nname, r.weight AS weight "
                    "ORDER BY r.weight DESC "
                    "LIMIT $limit",
                    video_id=video_id,
                    key=key,
                    limit=limit,
                )
            )
            for record in records:
                nkey = record.get("nkey")
                weight = record.get("weight") or 0
                if not nkey:
                    continue
                weights[nkey] = weights.get(nkey, 0) + float(weight)
    return weights


def compare_videos_detailed(
    video_id_a: str,
    video_id_b: str,
    window_k: int = 3,
    topic_jump_th: float = 0.05,
    max_new: int = 40,
    max_occurrences: int = 4,
    use_llm_prereq: bool = True,
) -> Dict[str, Any]:
    concepts_a = get_video_concept_keys(video_id_a)
    concepts_b = get_video_concept_keys(video_id_b)
    keys_a = set(concepts_a.keys())
    keys_b = set(concepts_b.keys())

    shared = keys_a & keys_b
    union = keys_a | keys_b
    jaccard = float(len(shared) / len(union)) if union else 0.0

    max_idx_a = _get_max_chunk_index(video_id_a)
    end_start = max(max_idx_a - window_k + 1, 0)
    end_concepts = get_window_concepts(video_id_a, end_start, max_idx_a) if max_idx_a >= 0 else {}
    start_concepts = get_window_concepts(video_id_b, 0, max(window_k - 1, 0))

    end_keys = set(end_concepts.keys())
    start_keys = set(start_concepts.keys())
    window_union = end_keys | start_keys
    window_shared = end_keys & start_keys
    window_jaccard = float(len(window_shared) / len(window_union)) if window_union else 0.0
    topic_shift_flag = bool(window_union) and window_jaccard < topic_jump_th
    topic_shift_reason = (
        f"Low overlap between end of A and start of B (Jaccard={window_jaccard:.2f})."
        if topic_shift_flag
        else "End/start overlap above threshold."
    )

    new_keys = [key for key in keys_b - keys_a]
    new_items = []
    raw_scores: Dict[str, float] = {}
    for key in new_keys:
        mentions, first_idx = _get_concept_frequency(video_id_b, key)
        co_strength = _get_concept_co_strength(video_id_b, key)
        raw_scores[key] = mentions + 0.2 * co_strength
        occurrences = get_concept_occurrences(
            video_id_b, key, limit=max_occurrences
        )
        first_seen = occurrences[0] if occurrences else {"chunk_index": first_idx, "snippet": ""}
        new_items.append(
            {
                "concept": concepts_b.get(key, key),
                "concept_key": key,
                "first_seen": {
                    "video": "B",
                    "chunk_index": first_seen.get("chunk_index", first_idx),
                    "snippet": first_seen.get("snippet", ""),
                },
                "occurrences": occurrences,
                "importance": 0.0,
                "confidence": 0.0,
            }
        )

    max_raw = max(raw_scores.values()) if raw_scores else 0.0
    for item in new_items:
        key = item["concept_key"]
        score = raw_scores.get(key, 0.0)
        importance = score / max_raw if max_raw > 0 else 0.0
        item["importance"] = round(importance, 3)
        item["confidence"] = round(importance, 3)

    new_items.sort(key=lambda x: x["importance"], reverse=True)
    new_items = new_items[:max_new]

    first_idx_b = get_first_occurrence_map(video_id_b)
    relations_b = get_relations(video_id_b)
    relations_a = get_relations(video_id_a)
    prereq_labels = {"prerequisite_of", "depends_on"}

    prereq_gaps = []
    order_violations = []
    prereq_gap_map: Dict[str, Dict[str, Any]] = {}

    for rel in relations_b:
        if rel["rel"] not in prereq_labels:
            continue
        prereq_key = rel["skey"]
        advanced_key = rel["tkey"]
        prereq_first = first_idx_b.get(prereq_key)
        advanced_first = first_idx_b.get(advanced_key)
        prereq_missing_a = prereq_key not in keys_a
        order_violation = (
            prereq_first is not None
            and advanced_first is not None
            and prereq_first > advanced_first
        )
        if prereq_missing_a or order_violation:
            gap_key = f"{advanced_key}::{prereq_key}"
            evidence = rel.get("evidence") or ""
            occurrences = get_concept_occurrences(
                video_id_b, advanced_key, limit=2
            )
            entry = prereq_gap_map.get(
                gap_key,
                {
                    "advanced_concept": rel["tname"],
                    "missing_prerequisites": [],
                    "evidence_in_b": occurrences,
                    "why_gap": "",
                    "confidence": 0.6,
                },
            )
            entry["missing_prerequisites"].append(rel["sname"])
            if prereq_missing_a:
                entry["why_gap"] = "Prerequisite is missing in video A."
                entry["confidence"] = max(entry["confidence"], 0.65)
            if order_violation:
                entry["why_gap"] = "Prerequisite appears after the dependent concept in video B."
                entry["confidence"] = max(entry["confidence"], 0.7)
            if evidence and not entry.get("evidence_in_b"):
                entry["evidence_in_b"] = [{"chunk_index": advanced_first or -1, "snippet": evidence}]
            prereq_gap_map[gap_key] = entry

        if order_violation:
            order_violations.append(
                {
                    "prerequisite": rel["sname"],
                    "dependent": rel["tname"],
                    "prereq_first_chunk": prereq_first,
                    "dependent_first_chunk": advanced_first,
                    "evidence": rel.get("evidence") or "",
                    "confidence": max(0.6, rel.get("confidence", 0.5)),
                }
            )

    if use_llm_prereq and new_items:
        top_new = new_items[: min(6, len(new_items))]
        a_norm_keys = {key for key in keys_a}
        for item in top_new:
            first_seen = item.get("first_seen", {})
            snippet = first_seen.get("snippet", "")
            prereqs = _infer_prereqs_from_chunk(item["concept"], snippet)
            missing = []
            for prereq in prereqs:
                prereq_key = _normalize_concept(prereq)
                if prereq_key and prereq_key not in a_norm_keys:
                    missing.append(prereq)
            if missing:
                gap_key = f"{item['concept_key']}::llm"
                entry = prereq_gap_map.get(
                    gap_key,
                    {
                        "advanced_concept": item["concept"],
                        "missing_prerequisites": [],
                        "evidence_in_b": [],
                        "why_gap": "Assumed prerequisites are missing in video A.",
                        "confidence": 0.45,
                    },
                )
                entry["missing_prerequisites"].extend(missing)
                if snippet:
                    entry["evidence_in_b"] = [
                        {
                            "chunk_index": first_seen.get("chunk_index", -1),
                            "snippet": snippet,
                        }
                    ]
                prereq_gap_map[gap_key] = entry

    prereq_gaps = list(prereq_gap_map.values())
    for gap in prereq_gaps:
        gap["missing_prerequisites"] = sorted(
            {item for item in gap.get("missing_prerequisites", []) if item}
        )

    bridge_gaps = []
    if topic_shift_flag:
        neighbor_weights = _get_co_neighbors(video_id_b, list(start_keys))
        bridge_candidates = {
            key: weight
            for key, weight in neighbor_weights.items()
            if key in keys_a and key not in start_keys
        }
        sorted_bridges = sorted(
            bridge_candidates.items(), key=lambda x: x[1], reverse=True
        )
        missing_bridge_concepts = [
            concepts_a.get(key, key) for key, _ in sorted_bridges[:8]
        ]
        suggested_query = ""
        if not missing_bridge_concepts:
            top_start_names = [start_concepts[k] for k in list(start_keys)[:5]]
            if top_start_names:
                suggested_query = "Search for: " + ", ".join(top_start_names)
        bridge_gaps.append(
            {
                "from_video_end_topic": [end_concepts[k] for k in end_keys],
                "to_video_start_topic": [start_concepts[k] for k in start_keys],
                "missing_bridge_concepts": missing_bridge_concepts,
                "suggested_bridge_query": suggested_query,
            }
        )

    rel_map_a = {(rel["skey"], rel["rel"], rel["tkey"]): rel for rel in relations_a}
    rel_map_b = {(rel["skey"], rel["rel"], rel["tkey"]): rel for rel in relations_b}
    relation_mismatches = []

    conflicts = set()
    for rel in relations_a:
        if rel["rel"] != "prerequisite_of":
            continue
        reverse_key = (rel["tkey"], "prerequisite_of", rel["skey"])
        if reverse_key in rel_map_b:
            conflicts.add((rel["skey"], rel["tkey"]))
            rel_b = rel_map_b[reverse_key]
            relation_mismatches.append(
                {
                    "source": rel["sname"],
                    "relation": rel["rel"],
                    "target": rel["tname"],
                    "present_in": "conflict",
                    "evidence_a": rel.get("evidence", ""),
                    "evidence_b": rel_b.get("evidence", ""),
                    "confidence": 0.7,
                }
            )

    for key, rel in rel_map_a.items():
        if key in rel_map_b:
            continue
        if rel["rel"] == "prerequisite_of" and (rel["skey"], rel["tkey"]) in conflicts:
            continue
        relation_mismatches.append(
            {
                "source": rel["sname"],
                "relation": rel["rel"],
                "target": rel["tname"],
                "present_in": "A_only",
                "evidence_a": rel.get("evidence", ""),
                "evidence_b": "",
                "confidence": max(0.4, rel.get("confidence", 0.5)),
            }
        )

    for key, rel in rel_map_b.items():
        if key in rel_map_a:
            continue
        relation_mismatches.append(
            {
                "source": rel["sname"],
                "relation": rel["rel"],
                "target": rel["tname"],
                "present_in": "B_only",
                "evidence_a": "",
                "evidence_b": rel.get("evidence", ""),
                "confidence": max(0.4, rel.get("confidence", 0.5)),
            }
        )

    summary = {
        "concepts_a": len(keys_a),
        "concepts_b": len(keys_b),
        "shared": len(shared),
        "jaccard": round(jaccard, 3),
        "topic_shift_flag": topic_shift_flag,
        "topic_shift_reason": topic_shift_reason,
    }

    return {
        "summary": summary,
        "new_in_b": new_items,
        "prereq_gaps": prereq_gaps,
        "order_violations_b": order_violations,
        "bridge_gaps": bridge_gaps,
        "relation_mismatches": relation_mismatches,
    }


def debug_compare(video_id_a: str, video_id_b: str) -> None:
    result = compare_videos_detailed(video_id_a, video_id_b)
    print(json.dumps(result.get("summary", {}), indent=2))


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
    concepts_a = get_video_concept_keys(video_id_a)
    concepts_b = get_video_concept_keys(video_id_b)
    keys_a = set(concepts_a.keys())
    keys_b = set(concepts_b.keys())
    return {
        "only_in_a": sorted(concepts_a[key] for key in keys_a - keys_b),
        "only_in_b": sorted(concepts_b[key] for key in keys_b - keys_a),
        "shared": sorted(concepts_a[key] for key in keys_a & keys_b),
    }


def compare_with_reference(video_id: str, reference_terms: List[str]) -> Dict[str, List[str]]:
    concepts = set(get_video_concepts(video_id))
    normalized_ref = {term.strip() for term in reference_terms if term.strip()}
    return {
        "missing": sorted(normalized_ref - concepts),
        "covered": sorted(concepts & normalized_ref),
    }
