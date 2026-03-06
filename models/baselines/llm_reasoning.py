import json
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from models.base import BaseGapModel, PredictionResult
from dataset.utils import clean_json_text
from utility import compress_transcript_for_gaps, llm


class LLMReasoningBaseline(BaseGapModel):
    name = "llm_reasoning"

    def predict(self, video_id, transcript, chunks, curriculum) -> PredictionResult:
        concepts = [concept.name for concept in curriculum]
        condensed = compress_transcript_for_gaps(transcript)
        prompt = ChatPromptTemplate.from_template(
            """
            You are identifying missing prerequisites from a curriculum list.
            Return STRICT JSON only:
            {{"missing": ["concept1", "concept2"]}}
            Rules:
            - Choose only from the provided concept list.
            - Do not invent new terms.
            - Use transcript context only.

            Concepts:
            {concepts}

            Transcript:
            {transcript}
            """
        )
        response = (prompt | llm).invoke(
            {"concepts": ", ".join(concepts), "transcript": condensed}
        )
        cleaned = clean_json_text(response.content)
        try:
            payload = json.loads(cleaned)
            missing_names = payload.get("missing", [])
        except json.JSONDecodeError:
            missing_names = []

        missing = {}
        explained = {}
        missing_set = {name.lower().strip() for name in missing_names if name}
        for concept in curriculum:
            is_missing = concept.name.lower().strip() in missing_set
            explained[concept.concept_id] = 0.0 if is_missing else 1.0
            missing[concept.concept_id] = 1.0 if is_missing else 0.0
        return PredictionResult(explained=explained, missing=missing, evidence={})
