import csv
import json
import math
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from dataset.schema import ConceptCoverageLabel, CurriculumConcept, EvidenceItem, VideoItem
from dataset.utils import clean_json_text
from utility import explanation_score, is_intro_like, llm, _get_embedding_model


def _find_snippet(text: str, term: str, window: int = 120) -> str:
    lower = text.lower()
    target = term.lower()
    pos = lower.find(target)
    if pos == -1:
        return text[: window * 2]
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    return text[start:end]


def _mentions_any(text: str, aliases: List[str]) -> Optional[str]:
    lower = text.lower()
    for alias in aliases:
        target = alias.lower().strip()
        if not target:
            continue
        if target in lower:
            return target
    return None


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


class BaseLabeler:
    name = "base"

    def label(
        self,
        video: VideoItem,
        curriculum: List[CurriculumConcept],
        chunks: List[dict],
    ) -> List[ConceptCoverageLabel]:
        raise NotImplementedError


class HeuristicLabeler(BaseLabeler):
    name = "heuristic"

    def __init__(self, explanation_threshold: int = 2):
        self.explanation_threshold = explanation_threshold

    def label(
        self,
        video: VideoItem,
        curriculum: List[CurriculumConcept],
        chunks: List[dict],
    ) -> List[ConceptCoverageLabel]:
        labels: List[ConceptCoverageLabel] = []
        for concept in curriculum:
            aliases = [concept.name] + concept.aliases
            explained = False
            evidence_items: List[EvidenceItem] = []
            best_chunk_idx = -1
            for chunk in chunks:
                text = chunk["text"]
                idx = chunk["chunk_index"]
                match = _mentions_any(text, aliases)
                if not match:
                    continue
                snippet = _find_snippet(text, match)
                evidence_items.append(
                    EvidenceItem(
                        chunk_index=idx,
                        snippet=snippet,
                        reason="mention",
                    )
                )
                score = explanation_score(text)
                if idx <= 1 and is_intro_like(text) and score < self.explanation_threshold:
                    continue
                if score >= self.explanation_threshold:
                    explained = True
                    best_chunk_idx = idx
                    break
            label_value = 1 if explained else 0
            confidence = 0.7 if explained else 0.5
            labels.append(
                ConceptCoverageLabel(
                    video_id=video.video_id,
                    concept_id=concept.concept_id,
                    explained_label=label_value,
                    evidence=evidence_items[:3],
                    label_source=self.name,
                    label_confidence=confidence,
                    created_at=datetime.utcnow().isoformat(),
                )
            )
        return labels


class LLMJudgeLabeler(BaseLabeler):
    name = "llm_judge"

    def __init__(self, variant: str = "A", max_snippets: int = 3):
        self.variant = variant
        self.max_snippets = max_snippets
        self._chunk_texts: List[str] = []
        self._chunk_embeddings: List[List[float]] = []
        self._model = _get_embedding_model()

    def _ensure_embeddings(self, chunks: List[dict]) -> None:
        if self._chunk_embeddings and self._chunk_texts:
            return
        self._chunk_texts = [chunk["text"] for chunk in chunks]
        if not self._chunk_texts:
            return
        self._chunk_embeddings = self._model.embed_documents(self._chunk_texts)

    def _select_snippets(
        self,
        concept: CurriculumConcept,
        chunks: List[dict],
    ) -> List[Tuple[int, str]]:
        aliases = [concept.name] + concept.aliases
        snippets: List[Tuple[int, str]] = []
        for chunk in chunks:
            match = _mentions_any(chunk["text"], aliases)
            if match:
                snippets.append(
                    (chunk["chunk_index"], _find_snippet(chunk["text"], match))
                )
        if snippets:
            return snippets[: self.max_snippets]

        self._ensure_embeddings(chunks)
        if not self._chunk_embeddings:
            return []
        query = concept.name
        if concept.short_definition:
            query = f"{concept.name}: {concept.short_definition}"
        query_vec = self._model.embed_documents([query])[0]
        scored: List[Tuple[float, int]] = []
        for idx, vec in enumerate(self._chunk_embeddings):
            scored.append((_cosine_similarity(query_vec, vec), idx))
        scored.sort(reverse=True, key=lambda x: x[0])
        for _, idx in scored[: self.max_snippets]:
            snippets.append((idx, self._chunk_texts[idx][:240]))
        return snippets

    def _prompt(self, concept: CurriculumConcept, snippets: List[Tuple[int, str]]) -> str:
        snippet_text = "\n".join(
            f"- chunk {idx}: {text}" for idx, text in snippets
        ) or "None"
        style = "concise" if self.variant == "A" else "deliberate"
        prompt = ChatPromptTemplate.from_template(
            """
            You are a {style} judge of transcript coverage.
            Decide if the concept is explained in the snippets.
            Return STRICT JSON:
            {{"label": "explained|not_explained", "confidence": 0.0, "rationale": "...", "evidence_snippet": "..."}}
            Concept: {name}
            Definition: {definition}
            Snippets:
            {snippets}
            """
        )
        return prompt.format(
            style=style,
            name=concept.name,
            definition=concept.short_definition or "None",
            snippets=snippet_text,
        )

    def label(
        self,
        video: VideoItem,
        curriculum: List[CurriculumConcept],
        chunks: List[dict],
    ) -> List[ConceptCoverageLabel]:
        labels: List[ConceptCoverageLabel] = []
        for concept in curriculum:
            snippets = self._select_snippets(concept, chunks)
            prompt = self._prompt(concept, snippets)
            response = llm.invoke(prompt)
            cleaned = clean_json_text(response.content)
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                payload = {"label": "not_explained", "confidence": 0.3}
            label = payload.get("label", "not_explained")
            explained = 1 if label == "explained" else 0
            confidence = payload.get("confidence", 0.5)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.5
            evidence = []
            evidence_snippet = payload.get("evidence_snippet") or ""
            if evidence_snippet and snippets:
                evidence.append(
                    EvidenceItem(
                        chunk_index=snippets[0][0],
                        snippet=evidence_snippet,
                        reason="llm",
                    )
                )
            labels.append(
                ConceptCoverageLabel(
                    video_id=video.video_id,
                    concept_id=concept.concept_id,
                    explained_label=explained,
                    evidence=evidence,
                    label_source=self.name,
                    label_confidence=confidence,
                    created_at=datetime.utcnow().isoformat(),
                )
            )
        return labels


class AgreementLabeler(BaseLabeler):
    name = "llm_agreement"

    def __init__(self):
        self.judge_a = LLMJudgeLabeler(variant="A")
        self.judge_b = LLMJudgeLabeler(variant="B")

    def label(
        self,
        video: VideoItem,
        curriculum: List[CurriculumConcept],
        chunks: List[dict],
    ) -> List[ConceptCoverageLabel]:
        labels_a = self.judge_a.label(video, curriculum, chunks)
        labels_b = self.judge_b.label(video, curriculum, chunks)
        by_id: Dict[str, ConceptCoverageLabel] = {
            label.concept_id: label for label in labels_a
        }
        results: List[ConceptCoverageLabel] = []
        for label_b in labels_b:
            label_a = by_id.get(label_b.concept_id)
            if not label_a:
                results.append(label_b)
                continue
            if label_a.explained_label == label_b.explained_label:
                label_a.label_source = self.name
                label_a.label_confidence = min(
                    1.0, (label_a.label_confidence + label_b.label_confidence) / 2
                )
                results.append(label_a)
            else:
                results.append(
                    ConceptCoverageLabel(
                        video_id=video.video_id,
                        concept_id=label_b.concept_id,
                        explained_label=-1,
                        evidence=[],
                        label_source="needs_review",
                        label_confidence=0.0,
                        created_at=datetime.utcnow().isoformat(),
                    )
                )
        return results


def export_disagreements(
    labels: Iterable[ConceptCoverageLabel], csv_path: str
) -> None:
    rows = [
        label
        for label in labels
        if label.label_source == "needs_review" or label.explained_label == -1
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "concept_id", "label_source"])
        for label in rows:
            writer.writerow(
                [label.video_id, label.concept_id, label.label_source]
            )
