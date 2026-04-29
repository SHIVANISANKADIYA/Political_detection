import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from deep_translator import GoogleTranslator

# Keep your current label set compatible with config.json
# Current model labels are: O, B/I-PARTY, B/I-POL_LEADER, B/I-SCHEME
# If any ORG exists in data, we map it to PARTY so the head size stays unchanged.

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "multilingual_training_data.jsonl"

SOURCE_FILES = [
    DATA_DIR / "labelled_training_data_batch1.jsonl",
    DATA_DIR / "labelled_training_data_batch2.jsonl",
]

TARGET_LANGS = ["hi", "ta", "te"]  # Hindi, Tamil, Telugu


def normalize_label(label: str) -> str:
    label = (label or "").strip().upper()
    if label == "ORG":
        return "PARTY"
    return label


def load_examples(files: List[Path]) -> List[dict]:
    examples = []
    for fp in files:
        if not fp.exists():
            continue
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip().rstrip(",")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sent = obj.get("sentence") or obj.get("text") or ""
                    ents = obj.get("entities", [])
                    if sent and isinstance(ents, list):
                        # normalize labels
                        new_ents = []
                        for e in ents:
                            text = (e.get("text") or "").strip()
                            label = normalize_label(e.get("label") or "")
                            if text and label:
                                new_ents.append({"text": text, "label": label})
                        obj["sentence"] = sent
                        obj["entities"] = new_ents
                        examples.append(obj)
                except Exception:
                    print(f"Skipping invalid JSON in {fp.name} line {line_no}")
    return examples


def replace_entities_with_placeholders(sentence: str, entities: List[dict]) -> Tuple[str, Dict[str, str]]:
    """
    Replace each entity occurrence with a placeholder so translation doesn't destroy alignment.
    """
    placeholder_map = {}
    protected = sentence

    # replace longer entities first to avoid partial overlaps
    sorted_entities = sorted(entities, key=lambda x: len(x["text"]), reverse=True)

    for i, ent in enumerate(sorted_entities):
        ent_text = ent["text"]
        placeholder = f"__ENT_{i}__"
        # first exact occurrence only
        idx = protected.find(ent_text)
        if idx == -1:
            idx = protected.lower().find(ent_text.lower())
        if idx == -1:
            continue
        protected = protected[:idx] + placeholder + protected[idx + len(ent_text):]
        placeholder_map[placeholder] = ent_text

    return protected, placeholder_map


def restore_placeholders(text: str, placeholder_map: Dict[str, str]) -> str:
    restored = text
    for ph, ent_text in placeholder_map.items():
        restored = restored.replace(ph, ent_text)
    return restored


def translate_text(text: str, target_lang: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text


def rebuild_entities(sentence: str, entities: List[dict]) -> List[dict]:
    """
    After translation + placeholder restoration, locate entity spans again.
    """
    rebuilt = []
    used_spans = []

    for ent in entities:
        ent_text = ent["text"]
        label = normalize_label(ent["label"])

        # find a non-overlapping occurrence
        start = sentence.find(ent_text)
        while start != -1:
            end = start + len(ent_text)
            overlap = any(not (end <= s or start >= e) for s, e in used_spans)
            if not overlap:
                used_spans.append((start, end))
                rebuilt.append({"text": ent_text, "label": label})
                break
            start = sentence.find(ent_text, start + 1)

    return rebuilt


def augment_examples(examples: List[dict]) -> List[dict]:
    out = []

    for ex in examples:
        sentence = ex["sentence"].strip()
        entities = ex.get("entities", [])

        if not sentence:
            continue

        # original
        out.append({"sentence": sentence, "entities": entities})

        # translated versions
        for lang in TARGET_LANGS:
            protected, placeholder_map = replace_entities_with_placeholders(sentence, entities)
            translated = translate_text(protected, lang)
            restored = restore_placeholders(translated, placeholder_map)
            rebuilt_entities = rebuild_entities(restored, entities)

            # keep only if sentence is non-empty
            if restored.strip():
                out.append({
                    "sentence": restored.strip(),
                    "entities": rebuilt_entities
                })

            time.sleep(0.15)

    return out


def main():
    examples = load_examples(SOURCE_FILES)
    if not examples:
        raise FileNotFoundError("No labeled examples found in data/labelled_training_data_batch1.jsonl or batch2.")

    augmented = augment_examples(examples)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for obj in augmented:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {len(augmented)} examples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()