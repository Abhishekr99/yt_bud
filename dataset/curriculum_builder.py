import json
from datetime import datetime
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from dataset.schema import CurriculumConcept
from dataset.utils import clean_json_text, slugify
from utility import llm


def build_curriculum(
    domain: str, topic: str, curriculum_id: str, max_concepts: int = 40
) -> dict:
    prompt = ChatPromptTemplate.from_template(
        """
        You are building a curriculum concept list for a topic.
        DO NOT use any transcript text.
        Return STRICT JSON only:
        {{
          "concepts": [
            {{
              "name": "...",
              "aliases": ["..."],
              "short_definition": "...",
              "prerequisites": ["prereq name", "..."]
            }}
          ]
        }}
        Rules:
        - Provide up to {max_concepts} core concepts for the topic.
        - Keep names concise (1-4 words).
        - Prerequisites should refer to other concept names in the list.
        Domain: {domain}
        Topic: {topic}
        """
    )
    response = (prompt | llm).invoke(
        {"domain": domain, "topic": topic, "max_concepts": max_concepts}
    )
    cleaned = clean_json_text(response.content)
    payload = json.loads(cleaned)
    concepts_raw = payload.get("concepts", [])

    concepts: List[CurriculumConcept] = []
    name_to_id = {}
    for item in concepts_raw:
        name = (item.get("name") or "").strip()
        if not name:
            continue
        concept_id = slugify(name)
        name_to_id[name.lower()] = concept_id
        concepts.append(
            CurriculumConcept(
                concept_id=concept_id,
                name=name,
                aliases=item.get("aliases", []) or [],
                short_definition=item.get("short_definition"),
                prerequisites=[],
            )
        )

    for concept in concepts:
        raw = next(
            (item for item in concepts_raw if (item.get("name") or "").strip() == concept.name),
            None,
        )
        prereqs = []
        if raw:
            for prereq in raw.get("prerequisites", []) or []:
                prereq_id = name_to_id.get(str(prereq).lower().strip())
                if prereq_id and prereq_id != concept.concept_id:
                    prereqs.append(prereq_id)
        concept.prerequisites = sorted(set(prereqs))

    return {
        "curriculum_id": curriculum_id,
        "domain": domain,
        "topic": topic,
        "created_at": datetime.utcnow().isoformat(),
        "concepts": [concept.to_dict() for concept in concepts],
    }
