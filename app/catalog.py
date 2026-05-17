"""SHL catalog loader. Pure data — no LLM / embedding calls here."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KEYS_TO_LETTER = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


@dataclass
class Product:
    entity_id: str
    name: str
    url: str
    description: str
    keys: list[str]
    job_levels: list[str]
    languages: list[str]
    duration: str
    remote: str
    adaptive: str
    test_type: str  # derived: comma-joined letters from keys, e.g. "C, K"

    def search_text(self) -> str:
        """Concatenated text used for BM25 + dense embedding."""
        parts = [
            self.name,
            self.description,
        ]
        if self.keys:
            parts.append("Categories: " + ", ".join(self.keys))
        if self.job_levels:
            parts.append("Job levels: " + ", ".join(self.job_levels))
        if self.duration:
            parts.append("Duration: " + self.duration)
        if self.adaptive == "yes":
            parts.append("Adaptive test")
        return " | ".join(p for p in parts if p)


@dataclass
class Catalog:
    products: list[Product]
    by_id: dict[str, Product] = field(default_factory=dict)
    by_url: dict[str, Product] = field(default_factory=dict)
    by_name_lower: dict[str, Product] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for p in self.products:
            self.by_id[p.entity_id] = p
            self.by_url[p.url] = p
            self.by_name_lower[p.name.lower()] = p

    def find_by_name(self, name: str) -> Product | None:
        return self.by_name_lower.get(name.lower().strip())

    def is_known_url(self, url: str) -> bool:
        return url in self.by_url


def _derive_test_type(keys: list[str]) -> str:
    letters: list[str] = []
    for k in keys or []:
        letter = KEYS_TO_LETTER.get(k)
        if letter and letter not in letters:
            letters.append(letter)
    return ", ".join(letters)


def load_catalog(path: str | Path) -> Catalog:
    """Load the scraped SHL catalog JSON and build a Catalog object.

    Uses strict=False because the scraped JSON contains some control characters
    in description fields that strict mode rejects.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"), strict=False)
    products: list[Product] = []
    for d in raw:
        if d.get("status") != "ok":
            continue
        keys = d.get("keys") or []
        products.append(
            Product(
                entity_id=str(d.get("entity_id", "")),
                name=(d.get("name") or "").strip(),
                url=(d.get("link") or "").strip(),
                description=(d.get("description") or "").strip(),
                keys=keys,
                job_levels=d.get("job_levels") or [],
                languages=d.get("languages") or [],
                duration=(d.get("duration") or "").strip(),
                remote=(d.get("remote") or "").strip(),
                adaptive=(d.get("adaptive") or "").strip(),
                test_type=_derive_test_type(keys),
            )
        )
    return Catalog(products=products)
