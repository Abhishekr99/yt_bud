from datetime import datetime
from typing import Iterable, List

from dataset.schema import CurriculumConcept
from dataset.utils import slugify


def build_curriculum_from_list(
    concepts: Iterable[str],
    curriculum_id: str,
    domain: str,
    topic: str,
) -> dict:
    items: List[CurriculumConcept] = []
    for concept in concepts:
        name = concept.strip()
        if not name:
            continue
        items.append(
            CurriculumConcept(
                concept_id=slugify(name),
                name=name,
                aliases=[],
                short_definition=None,
                prerequisites=[],
            )
        )
    return {
        "curriculum_id": curriculum_id,
        "domain": domain,
        "topic": topic,
        "created_at": datetime.utcnow().isoformat(),
        "concepts": [item.to_dict() for item in items],
    }


def load_plaintext(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]
