import json
import re
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from utility import compress_transcript_for_topics, llm


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


def evaluate_rag_answers(
    question: str,
    transcript: str,
    vanilla_answer: str,
    graph_answer: str,
) -> Dict[str, Any]:
    condensed = compress_transcript_for_topics(transcript)
    prompt = ChatPromptTemplate.from_template(
        """
        You are an impartial judge comparing two answers to the same question.
        Score each answer on a 1-5 scale and provide brief notes.
        Metrics:
        - groundedness: stays within transcript evidence.
        - relevance: addresses the question directly.
        - completeness: covers the key points from the transcript.
        - clarity: concise and easy to follow.
        Return STRICT JSON only:
        {{
          "vanilla": {{
            "groundedness": {{"score": int, "notes": "..."}},
            "relevance": {{"score": int, "notes": "..."}},
            "completeness": {{"score": int, "notes": "..."}},
            "clarity": {{"score": int, "notes": "..."}}
          }},
          "graph": {{
            "groundedness": {{"score": int, "notes": "..."}},
            "relevance": {{"score": int, "notes": "..."}},
            "completeness": {{"score": int, "notes": "..."}},
            "clarity": {{"score": int, "notes": "..."}}
          }},
          "winner": "vanilla|graph|tie",
          "rationale": "short reason"
        }}

        Condensed Transcript:
        {transcript}

        Question:
        {question}

        Vanilla RAG Answer:
        {vanilla_answer}

        Graph RAG Answer:
        {graph_answer}
        """
    )
    response = (prompt | llm).invoke(
        {
            "transcript": condensed,
            "question": question,
            "vanilla_answer": vanilla_answer,
            "graph_answer": graph_answer,
        }
    )
    cleaned = _clean_json_text(response.content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_response": response.content}
