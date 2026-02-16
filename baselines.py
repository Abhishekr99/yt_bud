from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utility import embedding_similarity


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        return 0.0
    if matrix.shape[1] == 0:
        return 0.0
    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return float(sim)


def embedding_cosine_similarity_full(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    sim = embedding_similarity([text_a], [text_b])
    if sim < 0:
        return 0.0
    return float(sim)
