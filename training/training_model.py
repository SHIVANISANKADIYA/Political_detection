# training_model.py
import json
from pathlib import Path
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, BertConfig
import torch

# ---------- settings ----------
MODEL_DIR = Path("political-bert")   # where to load/save
BASE_PRETRAINED = "bert-base-multilingual-cased"
TRAIN_FILE = Path("political.jsonl")  # expected labeled file (one JSON per line)
OUTPUT_DIR = Path("./results_continue")
# ------------------------------

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"{TRAIN_FILE} not found. Please prepare labeled data in this JSONL file.")

# 1) load training data
train_examples = []
with TRAIN_FILE.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            # expect keys: "sentence" and "entities": [{"text":..,"label":..}, ...]
            if "sentence" in obj:
                train_examples.append(obj)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON at line {i}: {e}")

if not train_examples:
    raise ValueError("No valid training examples found in political.jsonl")

# 2) build label list from data (B- / I- style + O)
labels = set()
for ex in train_examples:
    for ent in ex.get("entities", []):
        lab = ent.get("label")
        if lab:
            labels.add("B-" + lab)
            labels.add("I-" + lab)
labels.add("O")
labels = sorted(labels)  # deterministic order

label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for lab, i in label2id.items()}

print("Labels:", labels)

# 3) tokenizer + model init
tokenizer = BertTokenizerFast.from_pretrained(BASE_PRETRAINED)
# If an existing fine-tuned model exists, load its weights and adapt; else init a new head.
if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
    print(f"Loading existing model from {MODEL_DIR}")
    model = BertForTokenClassification.from_pretrained(str(MODEL_DIR))
    # ensure label mapping matches; if not, we will override config for training
    # but safe route: set config label2id/id2label if absent
    cfg = model.config
    if not getattr(cfg, "label2id", None):
        cfg.label2id = label2id
        cfg.id2label = id2label
else:
    print("Initializing new model from base pretrained (with fresh classification head).")
    config = BertConfig.from_pretrained(BASE_PRETRAINED, num_labels=len(labels))
    config.label2id = label2id
    config.id2label = id2label
    model = BertForTokenClassification.from_pretrained(BASE_PRETRAINED, config=config)

# 4) tokenization + align labels
def tokenize_and_align_labels(example):
    sentence = example["sentence"]
    ents = example.get("entities", [])
    # tokenizer returns offset mapping for alignment
    enc = tokenizer(sentence, return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = enc["offset_mapping"]
    labels_for_tokens = ["O"] * len(enc["input_ids"])

    # for each entity, find char span and mark tokens
    for ent in ents:
        ent_text = ent.get("text", "")
        ent_label = ent.get("label", "")
        if not ent_text or not ent_label:
            continue
        start_char = sentence.find(ent_text)
        if start_char == -1:
            # fallback: try case-insensitive
            start_char = sentence.lower().find(ent_text.lower())
        if start_char == -1:
            # could not locate substring; skip (user should ensure exact substring)
            continue
        end_char = start_char + len(ent_text)
        token_indices = []
        for i, (s, e) in enumerate(offsets):
            # skip special tokens which may be (0,0)
            if s == 0 and e == 0:
                continue
            # overlap test
            if not (e <= start_char or s >= end_char):
                token_indices.append(i)
        if not token_indices:
            continue
        labels_for_tokens[token_indices[0]] = "B-" + ent_label
        for idx in token_indices[1:]:
            labels_for_tokens[idx] = "I-" + ent_label

    # convert label strings to ids (use label2id from model config)
    cfg_label2id = getattr(model.config, "label2id", {})
    label_ids = [cfg_label2id.get(l, cfg_label2id.get("O")) for l in labels_for_tokens]

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": label_ids
    }

# tokenize all
encodings = [tokenize_and_align_labels(ex) for ex in train_examples]

# build torch dataset
class NewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
    def __len__(self):
        return len(self.encodings)

dataset = NewDataset(encodings)

# 5) Trainer
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=False,
    remove_unused_columns=False,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# 6) train
trainer.train()

# 7) save
MODEL_DIR.mkdir(exist_ok=True)
model.save_pretrained(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
print("Saved fine-tuned model to", MODEL_DIR)