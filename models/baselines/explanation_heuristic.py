from typing import List

from models.base import BaseGapModel, PredictionResult
from utility import explanation_score, is_intro_like


class ExplanationHeuristicBaseline(BaseGapModel):
    name = "explanation_heuristic"

    def __init__(self, explanation_threshold: int = 2):
        self.explanation_threshold = explanation_threshold

    def predict(self, video_id, transcript, chunks, curriculum) -> PredictionResult:
        explained = {}
        missing = {}
        evidence = {}
        for concept in curriculum:
            aliases: List[str] = [concept.name] + concept.aliases
            found = False
            for chunk in chunks:
                text = chunk["text"]
                idx = chunk["chunk_index"]
                lower = text.lower()
                if not any(alias.lower() in lower for alias in aliases if alias.strip()):
                    continue
                score = explanation_score(text)
                if idx <= 1 and is_intro_like(text) and score < self.explanation_threshold:
                    continue
                if score >= self.explanation_threshold:
                    found = True
                    evidence[concept.concept_id] = [
                        {"chunk_index": idx, "snippet": text[:240]}
                    ]
                    break
            explained[concept.concept_id] = 1.0 if found else 0.0
            missing[concept.concept_id] = 0.0 if found else 1.0
        return PredictionResult(explained=explained, missing=missing, evidence=evidence)
