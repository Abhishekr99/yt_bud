from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.base import BaseGapModel, PredictionResult


class TFIDFDefinitionBaseline(BaseGapModel):
    name = "tfidf_definition"

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

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

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        for concept in curriculum:
            query = concept.name
            if concept.short_definition:
                query = f"{concept.name}: {concept.short_definition}"
            query_vec = vectorizer.transform([query])
            sims = cosine_similarity(query_vec, matrix)[0]
            best_idx = int(sims.argmax())
            best_score = float(sims[best_idx])
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
