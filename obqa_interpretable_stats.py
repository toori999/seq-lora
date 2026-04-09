from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset


CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "plants": [
        "plant", "plants", "tree", "trees", "flower", "flowers", "seed", "seeds",
        "leaf", "leaves", "root", "roots", "stem", "stems", "grass",
        "photosynthesis", "sprout", "sprouting", "bloom", "blooming", "pollen",
        "pollination", "crop", "crops",
    ],
    "animals": [
        "animal", "animals", "mammal", "mammals", "bird", "birds", "fish", "insect",
        "insects", "reptile", "reptiles", "amphibian", "amphibians", "frog", "frogs",
        "dog", "dogs", "cat", "cats", "bear", "bears", "wolf", "wolves", "deer",
        "squirrel", "squirrels", "hawk", "hawks", "owl", "owls", "spider",
        "spiders", "octopus", "octopi", "lion", "lions", "fox", "foxes", "snail",
        "snails", "geese", "goose", "mouse", "mice", "ram", "giraffe", "giraffes",
        "cheetah", "cheetahs", "hatch", "hibernate", "migration",
    ],
    "ecology_life_cycles": [
        "ecosystem", "ecosystems", "habitat", "habitats", "predator", "predators",
        "prey", "population", "populations", "organism", "organisms", "offspring",
        "inherit", "inherited", "trait", "traits", "camouflage", "adaptation",
        "adaptations", "migrate", "migration", "food chain", "survival",
        "extinct", "endangered",
    ],
    "human_body_health": [
        "human", "humans", "body", "bodies", "brain", "heart", "lung", "lungs",
        "stomach", "muscle", "muscles", "bone", "bones", "blood", "skin", "teeth",
        "tooth", "digest", "digestion", "breathe", "breathing", "exercise",
        "disease", "illness", "healthy", "health", "vitamin", "germ", "germs",
        "bacteria", "virus", "doctor", "medicine",
    ],
    "food_nutrition": [
        "food", "foods", "eat", "eating", "nutrition", "nutrient", "nutrients",
        "protein", "carbohydrate", "carbohydrates", "fat", "fats", "sugar",
        "vitamin", "calorie", "calories", "fruit", "vegetable",
        "vegetables", "meat", "milk", "diet",
    ],
    "weather_climate": [
        "weather", "climate", "temperature", "rain", "rainfall", "snow", "storm",
        "storms", "wind", "winds", "cloud", "clouds", "humidity", "season",
        "seasons", "forecast", "tornado", "hurricane", "drought", "flood",
    ],
    "earth_water_rocks": [
        "earth", "soil", "rock", "rocks", "mineral", "minerals", "fossil", "fossils",
        "volcano", "volcanic", "erosion", "sediment", "sediments", "ocean", "river",
        "lake", "stream", "groundwater", "water cycle", "evaporation", "condensation",
        "precipitation", "landform",
    ],
    "space_sun_moon_stars": [
        "sun", "moon", "star", "stars", "planet", "planets", "solar system",
        "earth orbit", "orbit", "galaxy", "galaxies", "comet", "comets", "asteroid",
        "asteroids", "eclipse", "constellation", "satellite", "day and night",
    ],
    "optics_vision": [
        "mirror", "mirrors", "lens", "lenses", "telescope", "microscope",
        "flashlight", "visible", "vision", "see", "sight", "image", "images",
        "magnify", "magnifying", "light pollution", "transparent", "opaque",
        "translucent", "reflect", "reflection", "refract", "refraction",
        "shadow",
    ],
    "measurement_tools_units": [
        "measure", "measuring", "measurement", "meter", "meters", "kilometer",
        "kilometers", "centimeter", "centimeters", "ruler", "rulers", "thermometer",
        "thermometers", "seismometer", "barometer",
        "compass", "compasses", "degrees", "celsius", "fahrenheit",
    ],
    "motion_forces": [
        "force", "forces", "motion", "moving", "move", "moves", "speed", "velocity",
        "friction", "gravity", "push", "pull", "mass", "weight", "accelerate",
        "acceleration", "momentum", "swing", "slope", "balance", "balanced",
    ],
    "heat_light_sound": [
        "heat", "thermal", "temperature", "light", "sound", "loud", "louder",
        "quiet", "echo", "vibration", "vibrations", "wave", "waves", "reflect",
        "reflection", "refract", "refraction", "shadow", "transparent", "opaque",
        "conductor", "insulator",
    ],
    "electricity_magnetism": [
        "electric", "electricity", "battery", "batteries", "circuit", "circuits",
        "wire", "wires", "current", "magnet", "magnets", "magnetic", "electromagnet",
        "static electricity", "charge", "charged", "voltage",
    ],
    "materials_properties": [
        "material", "materials", "metal", "metals", "wood", "plastic", "glass",
        "rubber", "fabric", "paper", "hard", "soft", "rough", "smooth", "flexible",
        "rigid", "strong", "strength", "waterproof", "absorb", "absorbs", "absorbent",
        "dissolve", "soluble", "insoluble",
    ],
    "matter_changes_mixtures": [
        "matter", "solid", "liquid", "gas", "mixture", "mixtures", "solution",
        "solutions", "melt", "melting", "freeze", "freezing", "boil", "boiling",
        "evaporate", "evaporation", "condense", "condensation", "reaction", "react",
        "chemical change", "physical change", "acid", "acids", "base", "bases",
        "mercury",
    ],
    "geography_landforms": [
        "mountain", "mountains", "desert", "deserts", "valley", "valleys", "canyon",
        "canyons", "plateau", "plateaus", "plain", "plains", "island", "islands",
        "continent", "continents", "country", "countries", "state", "states",
        "hemisphere", "equator", "map", "maps", "landmark", "landmarks",
        "mount rushmore",
    ],
}


def _normalize_text(text: str) -> str:
    text = text.lower()
    for ch in [",", ".", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "/", "\\", "-", "_", "\"", "'"]:
        text = text.replace(ch, " ")
    text = " ".join(text.split())
    return f" {text} "


def _contains_phrase(norm_text: str, phrase: str) -> bool:
    phrase = " ".join(phrase.lower().split())
    return f" {phrase} " in norm_text


def _build_obqa_text(example: Dict) -> Tuple[str, str]:
    question = str(example.get("question_stem", "")).strip()
    choices = example.get("choices", {})
    choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
    joined_choices = " ".join(str(x).strip() for x in choice_texts)
    full_text = f"{question} {joined_choices}".strip()
    return question, full_text


def classify_obqa(example: Dict) -> Tuple[str, Dict[str, List[str]]]:
    question, full_text = _build_obqa_text(example)
    q_norm = _normalize_text(question)
    full_norm = _normalize_text(full_text)

    override_rules = [
        ("space_sun_moon_stars", [" daylight ", " solstice ", " equinox ", " hemisphere tilted ", " moon ", " moons ", " star ", " stars ", " sun "]),
        ("animals", [" squirrel ", " squirrels ", " hawk ", " hawks ", " owl ", " owls ", " spider ", " spiders ", " octopus ", " lion ", " lions ", " fox ", " foxes ", " snail ", " snails ", " geese ", " goose ", " mouse ", " mice "]),
        ("geography_landforms", [" mount rushmore ", " hemisphere ", " equator ", " continent ", " desert ", " canyon ", " plateau ", " mountain ", " map "]),
        ("optics_vision", [" telescope ", " microscope ", " flashlight ", " headlights ", " reflector ", " mirror ", " mirrors ", " lens ", " lenses ", " light pollution ", " visible ", " image ", " camouflage "]),
        ("earth_water_rocks", [" seismometer ", " earthquake ", " earthquakes ", " glacier ", " glaciers ", " mining ", " mine ", " mines "]),
        ("human_body_health", [" poison ", " poisonous ", " toxic ", " stomach ", " digest ", " digestion ", " waste ", " healthy ", " disease ", " illness ", " child ", " children ", " pharmacy ", " gestating ", " birthing "]),
        ("ecology_life_cycles", [" offspring ", " inherited ", " trait ", " traits ", " predator ", " predators ", " prey ", " organism ", " organisms ", " habitat ", " habitats ", " migrate ", " migration ", " hunted ", " hunting "]),
        ("measurement_tools_units", [" thermometer ", " thermometers ", " seismometer ", " measure ", " measuring ", " kilometer ", " kilometers ", " centimeter ", " centimeters ", " meter ", " meters ", " compass ", " degrees ", " fahrenheit ", " celsius "]),
        ("electricity_magnetism", [" conductor ", " conductors ", " outlet ", " outlets ", " plug ", " plugs ", " battery ", " batteries ", " remote device "]),
        ("materials_properties", [" reprocess ", " reuse ", " reusable ", " plastic bottle ", " recycle ", " recycling ", " arrows in a triangular shape "]),
        ("heat_light_sound", [" sonar ", " echo ", " echoes "]),
        ("matter_changes_mixtures", [" mercury ", " acid ", " acids ", " chemical change ", " physical change ", " gas ", " liquid ", " solid ", " mixture ", " solution ", " dissolved ", " freezer ", " freezes ", " freezing ", " water vapor "]),
    ]

    for cat, needles in override_rules:
        if any(needle in full_norm for needle in needles):
            return cat, {cat: [needle.strip() for needle in needles if needle in full_norm]}

    per_cat_hits: Dict[str, List[str]] = {}
    per_cat_score: Dict[str, int] = {}

    for cat, keywords in CATEGORY_KEYWORDS.items():
        hits: List[str] = []
        score = 0
        for kw in keywords:
            in_q = _contains_phrase(q_norm, kw)
            in_full = _contains_phrase(full_norm, kw)
            if not in_full:
                continue
            hits.append(kw)
            score += 2 if in_q else 1
        if hits:
            per_cat_hits[cat] = hits
            per_cat_score[cat] = score

    if not per_cat_score:
        return "other_unknown", {}

    # Prefer the category with the highest score; break ties by number of unique hits,
    # then lexicographically for determinism.
    best = sorted(
        per_cat_score.keys(),
        key=lambda cat: (-per_cat_score[cat], -len(set(per_cat_hits[cat])), cat),
    )[0]
    return best, per_cat_hits


def _load_obqa_split(split: str) -> Dataset:
    offline = os.environ.get("HF_DATASETS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"
    if not offline:
        try:
            return load_dataset("openbookqa", "main", split=split)
        except Exception:
            pass

    if offline:
        print("[Load] offline mode detected, using cached arrow files.")

    try:
        cache_root = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "huggingface",
            "datasets",
            "openbookqa",
            "main",
            "0.0.0",
            "388097ea7776314e93a529163e0fea805b8a6454",
        )
        file_map = {
            "train": "openbookqa-train.arrow",
            "validation": "openbookqa-validation.arrow",
            "test": "openbookqa-test.arrow",
        }
        fp = os.path.join(cache_root, file_map[split])
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        print(f"[Load] fallback to cached arrow -> {fp}")
        return Dataset.from_file(fp)
    except Exception:
        raise RuntimeError(
            "Failed to load OpenBookQA via load_dataset() and cached arrow fallback. "
            "If the dataset is cached locally, try running with HF_DATASETS_OFFLINE=1."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Heuristic interpretable category statistics for OpenBookQA.")
    ap.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    ap.add_argument("--save_csv", type=str, default="")
    ap.add_argument("--show_examples_per_cat", type=int, default=3)
    args = ap.parse_args()

    ds = _load_obqa_split(args.split)
    print(f"[Load] split={args.split} size={len(ds)}")
    print(f"[Info] available heuristic categories={len(CATEGORY_KEYWORDS)} + other_unknown")

    rows: List[Dict[str, str]] = []
    cat_counter: Counter = Counter()
    cat_examples: Dict[str, List[str]] = defaultdict(list)

    for ex in ds:
        category, hits = classify_obqa(ex)
        question, full_text = _build_obqa_text(ex)
        hit_terms = sorted(set(hits.get(category, [])))
        cat_counter[category] += 1
        if len(cat_examples[category]) < max(0, int(args.show_examples_per_cat)):
            cat_examples[category].append(question)
        rows.append(
            {
                "id": str(ex.get("id", "")),
                "question_stem": question,
                "primary_category": category,
                "matched_terms": "|".join(hit_terms),
                "full_text": full_text,
            }
        )

    total = len(rows)
    print("\n[Category Counts]")
    for cat, cnt in sorted(cat_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{cat:24s}  count={cnt:4d}  pct={100.0 * cnt / max(total, 1):6.2f}%")

    if int(args.show_examples_per_cat) > 0:
        print("\n[Examples]")
        for cat, _ in sorted(cat_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"\n## {cat}")
            for i, q in enumerate(cat_examples[cat], start=1):
                print(f"{i}. {q}")

    if args.save_csv:
        out_path = args.save_csv
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "question_stem", "primary_category", "matched_terms", "full_text"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Save] wrote per-example category assignments -> {out_path}")


if __name__ == "__main__":
    main()
