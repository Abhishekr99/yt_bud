from typing import List

from models.base import BaseGapModel, PredictionResult
class MentionOnlyBaseline(BaseGapModel):
    name = "mention_only"

    def predict(self, video_id, transcript, chunks, curriculum) -> PredictionResult:
        explained = {}
        missing = {}
        evidence = {}
        lower_text = transcript.lower()
        for concept in curriculum:
            aliases: List[str] = [concept.name] + concept.aliases
            found = False
            for alias in aliases:
                if alias.lower().strip() in lower_text:
                    found = True
                    break
            explained[concept.concept_id] = 1.0 if found else 0.0
            missing[concept.concept_id] = 0.0 if found else 1.0
            if found:
                evidence[concept.concept_id] = [{"snippet": alias}]
        return PredictionResult(explained=explained, missing=missing, evidence=evidence)
