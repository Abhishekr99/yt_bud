import math
from typing import List

from models.base import BaseGapModel, PredictionResult
from utility import _get_embedding_model


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class EmbeddingDefinitionBaseline(BaseGapModel):
    name = "embedding_definition"

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold
        self._model = _get_embedding_model()

    def predict(self, video_id, transcript, chunks, curriculum) -> PredictionResult:
        explained = {}
        missing = {}
        evidence = {}
        texts = [chunk["text"] for chunk in chunks]
        if not texts:
            for concept in curriculum:
                explained[concept.concept_id] = 0.0
                missing[concept.concept_id] = 1.0
            return PredictionResult(explained=explained, missing=missing, evidence=evidence)

        chunk_embeddings = self._model.embed_documents(texts)
        for concept in curriculum:
            query = concept.name
            if concept.short_definition:
                query = f"{concept.name}: {concept.short_definition}"
            query_vec = self._model.embed_documents([query])[0]
            scores = [
                _cosine_similarity(query_vec, vec) for vec in chunk_embeddings
            ]
            best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            best_score = float(scores[best_idx])
            is_explained = best_score >= self.threshold
            explained[concept.concept_id] = 1.0 if is_explained else 0.0
            missing[concept.concept_id] = 0.0 if is_explained else 1.0
            if is_explained:
                evidence[concept.concept_id] = [
                    {
                        "chunk_index": chunks[best_idx]["chunk_index"],
                        "snippet": chunks[best_idx]["text"][:240],
                        "score": round(best_score, 3),
                    }
                ]
        return PredictionResult(explained=explained, missing=missing, evidence=evidence)
