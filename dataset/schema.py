from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CurriculumConcept:
    concept_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    short_definition: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict) -> "CurriculumConcept":
        return cls(
            concept_id=payload["concept_id"],
            name=payload["name"],
            aliases=payload.get("aliases", []) or [],
            short_definition=payload.get("short_definition"),
            prerequisites=payload.get("prerequisites", []) or [],
        )

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "aliases": self.aliases,
            "short_definition": self.short_definition,
            "prerequisites": self.prerequisites,
        }


@dataclass
class VideoItem:
    video_id: str
    domain: str
    topic: str
    youtube_url: str
    language: str
    transcript_path: str
    curriculum_id: str

    @classmethod
    def from_dict(cls, payload: dict) -> "VideoItem":
        return cls(
            video_id=payload["video_id"],
            domain=payload.get("domain", ""),
            topic=payload.get("topic", ""),
            youtube_url=payload.get("youtube_url", ""),
            language=payload.get("language", "en"),
            transcript_path=payload.get("transcript_path", ""),
            curriculum_id=payload.get("curriculum_id", ""),
        )

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "domain": self.domain,
            "topic": self.topic,
            "youtube_url": self.youtube_url,
            "language": self.language,
            "transcript_path": self.transcript_path,
            "curriculum_id": self.curriculum_id,
        }


@dataclass
class EvidenceItem:
    chunk_index: int
    snippet: str
    reason: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "EvidenceItem":
        return cls(
            chunk_index=payload.get("chunk_index", -1),
            snippet=payload.get("snippet", ""),
            reason=payload.get("reason", ""),
        )

    def to_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "snippet": self.snippet,
            "reason": self.reason,
        }


@dataclass
class ConceptCoverageLabel:
    video_id: str
    concept_id: str
    explained_label: int
    evidence: List[EvidenceItem] = field(default_factory=list)
    label_source: str = "heuristic"
    label_confidence: float = 0.5
    created_at: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "ConceptCoverageLabel":
        evidence = [EvidenceItem.from_dict(item) for item in payload.get("evidence", [])]
        return cls(
            video_id=payload["video_id"],
            concept_id=payload["concept_id"],
            explained_label=payload.get("explained_label", -1),
            evidence=evidence,
            label_source=payload.get("label_source", "heuristic"),
            label_confidence=payload.get("label_confidence", 0.5),
            created_at=payload.get("created_at", ""),
        )

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "concept_id": self.concept_id,
            "explained_label": self.explained_label,
            "evidence": [item.to_dict() for item in self.evidence],
            "label_source": self.label_source,
            "label_confidence": self.label_confidence,
            "created_at": self.created_at,
        }
