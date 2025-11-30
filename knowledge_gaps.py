import json
import re
import urllib.parse
from typing import List, Dict, Any

import requests
from langchain_core.prompts import ChatPromptTemplate

from utility import llm


def identify_knowledge_gaps(transcript: str, max_gaps: int = 7) -> List[Dict[str, str]]:
    """
    Ask the LLM to spot potentially unfamiliar terms in the transcript.
    Returns a structured list of dictionaries with 'term', 'reason', and optional 'domain'.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You review a video transcript and flag terms the average viewer may not understand.

        Requirements (strict JSON only):
        - First, infer a short domain label (e.g., "finance", "physics", "history").
        - Then list up to {max_gaps} gap candidates mentioned but not explained.
        - Focus on proper nouns, acronyms, technical jargon, or niche concepts; skip generic words or anything already defined in the transcript.
        - Output EXACTLY this JSON shape with double quotes and no trailing text:
          {{"domain": "<domain>", "gaps": [{{"term": "<term1>", "reason": "<why it needs context>"}}, {{...}}]}}
        - Do not wrap the JSON in markdown fences.

        Transcript:
        {transcript}
        """
    )

    response = (prompt | llm).invoke({"transcript": transcript, "max_gaps": max_gaps})
    raw = response.content

    def _clean_json_text(text: str) -> str:
        """Remove code fences and isolate the JSON blob if present."""
        stripped = text.strip()
        # Drop Markdown fences if the model used them.
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z0-9]*", "", stripped, count=1).strip()
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()
        # Try to slice just the outer JSON object.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1:
            return stripped[start : end + 1]
        return stripped

    payload: Dict[str, Any] = {"domain": "general", "gaps": []}
    cleaned = _clean_json_text(raw)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Heuristic fallback: extract repeated "term"/"reason" pairs from the text.
        term_matches = re.findall(r'"?term"?\\s*[:=]\\s*"([^"]+)"', cleaned, flags=re.IGNORECASE)
        reason_matches = re.findall(r'"?reason"?\\s*[:=]\\s*"([^"]+)"', cleaned, flags=re.IGNORECASE)
        # Pair up terms and reasons in order; ignore trailing extras.
        for term, reason in zip(term_matches, reason_matches):
            payload["gaps"].append({"term": term.strip(), "reason": reason.strip() or "Needs context"})

    gaps = payload.get("gaps", [])
    domain = payload.get("domain", "general")
    return [{"term": g.get("term", "").strip(), "reason": g.get("reason", "").strip(), "domain": domain} for g in gaps if g.get("term")]


def _fetch_wikipedia_summary(term: str, sentences: int = 2) -> str:
    """
    Retrieve a concise summary for the term from Wikipedia.
    Uses the REST API to avoid HTML parsing; returns an empty string on failure.
    """
    encoded_term = urllib.parse.quote(term)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_term}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        extract = data.get("extract", "")
        if not extract:
            return ""
        # Keep only the first few sentences to avoid overwhelming the UI.
        return " ".join(extract.split(". ")[:sentences]).strip()
    except Exception:
        return ""


def _explain_with_llm(term: str, reason: str, domain: str) -> str:
    """
    Fallback: ask the LLM for a grounded, concise explanation when external retrieval fails.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Provide a concise, factual explanation for the term below.
        - Keep it under 3 sentences.
        - Match the domain context if provided.
        - Avoid speculation and keep a neutral tone.

        Term: {term}
        Domain: {domain}
        Why it matters in the transcript: {reason}
        """
    )
    response = (prompt | llm).invoke({"term": term, "reason": reason, "domain": domain})
    return response.content.strip()


def fetch_gap_contexts(gaps: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    For each gap term, pull a short explanation from Wikipedia or fall back to the LLM.
    Returns a list of {term, reason, context}.
    """
    enriched = []
    for gap in gaps:
        term = gap.get("term", "").strip()
        reason = gap.get("reason", "Needs context")
        domain = gap.get("domain", "general")
        if not term:
            continue

        summary = _fetch_wikipedia_summary(term)
        context = summary if summary else _explain_with_llm(term, reason, domain)
        enriched.append({"term": term, "reason": reason, "context": context})
    return enriched


def generate_enriched_notes(transcript: str, gap_contexts: List[Dict[str, str]]) -> str:
    """
    Produce notes that weave in the newly fetched context.
    Only the top few gap explanations are used to avoid clutter.
    """
    context_snippets = "\n".join(
        f"- {item['term']}: {item['context']}" for item in gap_contexts[:5]
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You create concise, structured notes for a YouTube transcript enriched with background info.

        Inputs:
        - Transcript (verbatim): {transcript}
        - Gap Explanations (brief, authoritative): 
        {context_snippets}

        Instructions:
        - Write bullet-point notes with short sentences.
        - Integrate the gap explanations naturally the first time each term appears.
        - Keep the tone factual and neutral; do not add new claims.
        - Highlight 3-5 key enriched takeaways near the top.
        - Stay concise; avoid repeating the full explanations verbatim.
        """
    )

    response = (prompt | llm).invoke(
        {"transcript": transcript, "context_snippets": context_snippets}
    )
    return response.content.strip()
