from typing import Dict, List

from graph_rag import get_concept_occurrences, get_relations, get_video_concept_keys
from models.base import BaseGapModel, PredictionResult
from utility import explanation_score, is_intro_like, normalize_concept


class Neo4jGraphGapModel(BaseGapModel):
    name = "neo4j_graph"

    def __init__(self, explanation_threshold: int = 2):
        self.explanation_threshold = explanation_threshold

    def _map_concepts(self, video_id: str, curriculum) -> Dict[str, str]:
        graph_concepts = get_video_concept_keys(video_id)
        mapping: Dict[str, str] = {}
        for concept in curriculum:
            aliases = [concept.name] + concept.aliases
            match_key = ""
            for alias in aliases:
                key = normalize_concept(alias)
                if key in graph_concepts:
                    match_key = key
                    break
            mapping[concept.concept_id] = match_key
        return mapping

    def _has_definitional_relation(self, video_id: str, key: str) -> bool:
        for rel in get_relations(video_id):
            if rel.get("tkey") == key and rel.get("rel") in {"defines", "explains"}:
                return True
        return False

    def _explained_score(self, video_id: str, key: str, name: str) -> float:
        if not key:
            return 0.0
        occurrences = get_concept_occurrences(video_id, key, concept_name=name, limit=6)
        if not occurrences:
            return 0.0
        explained = False
        for occ in occurrences:
            text = occ.get("text", "")
            idx = occ.get("chunk_index", 0)
            score = explanation_score(text)
            if idx <= 1 and is_intro_like(text) and score < self.explanation_threshold:
                continue
            if score >= self.explanation_threshold:
                explained = True
                break
        score = 0.2 + min(0.5, 0.1 * len(occurrences))
        if explained:
            score += 0.4
        if self._has_definitional_relation(video_id, key):
            score += 0.2
        return min(score, 1.0)

    def predict(self, video_id, transcript, chunks, curriculum) -> PredictionResult:
        explained: Dict[str, float] = {}
        missing: Dict[str, float] = {}
        evidence: Dict[str, List[dict]] = {}
        mapping = self._map_concepts(video_id, curriculum)

        for concept in curriculum:
            key = mapping.get(concept.concept_id, "")
            score = self._explained_score(video_id, key, concept.name)
            explained[concept.concept_id] = score
            missing[concept.concept_id] = 1.0 - score
            if score > 0:
                evidence[concept.concept_id] = [{"concept_key": key, "score": score}]

        for concept in curriculum:
            if explained.get(concept.concept_id, 0.0) < 0.5:
                continue
            for prereq_id in concept.prerequisites:
                if explained.get(prereq_id, 0.0) < 0.5:
                    missing[prereq_id] = max(missing.get(prereq_id, 0.0), 1.0)

        return PredictionResult(explained=explained, missing=missing, evidence=evidence)
