from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PredictionResult:
    explained: Dict[str, float] = field(default_factory=dict)
    missing: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, List[dict]] = field(default_factory=dict)


class BaseGapModel:
    name = "base"

    def predict(self, video_id, transcript, chunks, curriculum) -> PredictionResult:
        raise NotImplementedError
