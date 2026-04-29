# -*- coding: utf-8 -*-

import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import torch
import wikipedia
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForTokenClassification


# =========================================================
# 1) Load model and tokenizer
# =========================================================
MODEL_PATH = Path("political-bert")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading local model from: {MODEL_PATH.resolve()}")

# Use slow tokenizer from vocab.txt to avoid fast-tokenizer conversion issues
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# Load local fine-tuned model
model = BertForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.to(device)
model.eval()

id2label = {int(k): v for k, v in model.config.id2label.items()}


# =========================================================
# 2) Large test data
# Format:
#   (sentence, [(entity_text, label), ...])
# Labels must match your model:
#   PARTY, POL_LEADER, SCHEME
# =========================================================
test_data: List[Tuple[str, List[Tuple[str, str]]]] = [
    # ---------------- ENGLISH ----------------
    (
        "PM Modi addressed a public rally in Varanasi.",
        [("PM Modi", "POL_LEADER")]
    ),
    (
        "Rahul Gandhi visited Kerala to meet party workers.",
        [("Rahul Gandhi", "POL_LEADER")]
    ),
    (
        "Amit Shah inaugurated a new BJP office in Lucknow.",
        [("Amit Shah", "POL_LEADER"), ("BJP", "PARTY")]
    ),
    (
        "Sonia Gandhi chaired a Congress Working Committee meeting in Delhi.",
        [("Sonia Gandhi", "POL_LEADER"), ("Congress", "PARTY")]
    ),
    (
        "Mamata Banerjee announced new welfare schemes in Kolkata.",
        [("Mamata Banerjee", "POL_LEADER")]
    ),
    (
        "Arvind Kejriwal launched an education initiative in Delhi.",
        [("Arvind Kejriwal", "POL_LEADER")]
    ),
    (
        "Yogi Adityanath reviewed law and order in Uttar Pradesh.",
        [("Yogi Adityanath", "POL_LEADER")]
    ),
    (
        "K Chandrashekar Rao met the Governor at Raj Bhavan in Hyderabad.",
        [("K Chandrashekar Rao", "POL_LEADER")]
    ),
    (
        "Sharad Pawar attended an NCP meeting in Pune.",
        [("Sharad Pawar", "POL_LEADER"), ("NCP", "PARTY")]
    ),
    (
        "Nitish Kumar met the Prime Minister to discuss flood relief.",
        [("Nitish Kumar", "POL_LEADER")]
    ),
    (
        "Naveen Patnaik inaugurated a health center in Bhubaneswar.",
        [("Naveen Patnaik", "POL_LEADER")]
    ),
    (
        "Pinarayi Vijayan launched a digital literacy program in Kochi.",
        [("Pinarayi Vijayan", "POL_LEADER")]
    ),
    (
        "Uddhav Thackeray addressed Shiv Sena workers in Mumbai.",
        [("Uddhav Thackeray", "POL_LEADER"), ("Shiv Sena", "PARTY")]
    ),
    (
        "Devendra Fadnavis met industrialists to discuss investments.",
        [("Devendra Fadnavis", "POL_LEADER")]
    ),
    (
        "Himanta Biswa Sarma held talks with Assam civil officials.",
        [("Himanta Biswa Sarma", "POL_LEADER")]
    ),
    (
        "Bhupesh Baghel inaugurated an agricultural fair in Raipur.",
        [("Bhupesh Baghel", "POL_LEADER")]
    ),
    (
        "Ashok Gehlot announced a new pension scheme in Jaipur.",
        [("Ashok Gehlot", "POL_LEADER")]
    ),
    (
        "Hemant Soren discussed tribal welfare in Ranchi.",
        [("Hemant Soren", "POL_LEADER")]
    ),
    (
        "Basavaraj Bommai inaugurated a tech park in Bengaluru.",
        [("Basavaraj Bommai", "POL_LEADER")]
    ),
    (
        "M K Stalin met the Finance Minister in Chennai.",
        [("M K Stalin", "POL_LEADER")]
    ),
    (
        "Eknath Shinde chaired a cabinet meeting in Mumbai.",
        [("Eknath Shinde", "POL_LEADER")]
    ),
    (
        "Akhilesh Yadav criticized the government's economic policy.",
        [("Akhilesh Yadav", "POL_LEADER")]
    ),
    (
        "Mayawati addressed a BSP rally in Lucknow.",
        [("Mayawati", "POL_LEADER"), ("BSP", "PARTY")]
    ),
    (
        "Kamal Nath launched a youth employment program in Bhopal.",
        [("Kamal Nath", "POL_LEADER")]
    ),
    (
        "Manohar Lal Khattar inaugurated a sports complex in Gurugram.",
        [("Manohar Lal Khattar", "POL_LEADER")]
    ),
    (
        "Bhagwant Mann announced free electricity for farmers in Punjab.",
        [("Bhagwant Mann", "POL_LEADER")]
    ),
    (
        "Omar Abdullah called for peace talks in Jammu and Kashmir.",
        [("Omar Abdullah", "POL_LEADER")]
    ),
    (
        "Farooq Abdullah addressed party members in Srinagar.",
        [("Farooq Abdullah", "POL_LEADER")]
    ),
    (
        "Mehbooba Mufti met the Home Minister in Delhi.",
        [("Mehbooba Mufti", "POL_LEADER")]
    ),
    (
        "Jagan Mohan Reddy reviewed irrigation projects in Andhra Pradesh.",
        [("Jagan Mohan Reddy", "POL_LEADER")]
    ),
    (
        "Chandrababu Naidu announced digital initiatives in Vijayawada.",
        [("Chandrababu Naidu", "POL_LEADER")]
    ),
    (
        "KTR from TRS met business leaders in Hyderabad.",
        [("KTR", "POL_LEADER"), ("TRS", "PARTY")]
    ),
    (
        "Smriti Irani launched a women empowerment campaign in Amethi.",
        [("Smriti Irani", "POL_LEADER")]
    ),
    (
        "As JJP fades, Abhay Chautala eyes return, plans INLD rally in Jat heartland.",
        [("JJP", "PARTY"), ("Abhay Chautala", "POL_LEADER"), ("INLD", "PARTY")]
    ),
    (
        "As the Jannayak Janata Party (JJP) shrinks in Haryana, Indian National Lok Dal (INLD) chief Abhay Chautala is trying to reclaim the political space vacated by it.",
        [
            ("Jannayak Janata Party", "PARTY"),
            ("JJP", "PARTY"),
            ("Indian National Lok Dal", "PARTY"),
            ("INLD", "PARTY"),
            ("Abhay Chautala", "POL_LEADER"),
        ]
    ),
    (
        "On September 25, the birth anniversary of the late Devi Lal, Abhay will be holding a rally in Rohtak.",
        [("Devi Lal", "POL_LEADER"), ("Abhay", "POL_LEADER")]
    ),
    (
        "It is the first big event being held by the Chautala family since the death of former Haryana Chief Minister Om Prakash Chautala in December 2024.",
        [("Om Prakash Chautala", "POL_LEADER")]
    ),
    (
        "The site of the rally is also significant as Rohtak is the home turf of senior Congress leader and fellow Jat leader Bhupinder Singh Hooda.",
        [("Congress", "PARTY"), ("Bhupinder Singh Hooda", "POL_LEADER")]
    ),
    (
        "Abhay has invited Shiromani Akali Dal chief Sukhbir Singh Badal, Rajasthan MP Hanuman Beniwal, and National Conference leader Surinder Kumar Choudhary for the event.",
        [
            ("Abhay", "POL_LEADER"),
            ("Shiromani Akali Dal", "PARTY"),
            ("Sukhbir Singh Badal", "POL_LEADER"),
            ("Hanuman Beniwal", "POL_LEADER"),
            ("National Conference", "PARTY"),
            ("Surinder Kumar Choudhary", "POL_LEADER"),
        ]
    ),
    (
        "This is not a programme of the third front, and many of those leaders have already aligned themselves with either the BJP or the Congress.",
        [("BJP", "PARTY"), ("Congress", "PARTY")]
    ),
    (
        "The INLD leader has toured all 90 Assembly constituencies in Haryana to mobilise support for the rally.",
        [("INLD", "PARTY")]
    ),
    (
        "Abhay denied outright the suggestion that the choice of the venue was dictated by the fact that it is Hooda's turf.",
        [("Abhay", "POL_LEADER"), ("Hooda", "POL_LEADER")]
    ),
    (
        "Everyone in the country knows he is hand in glove with the BJP, he said.",
        [("BJP", "PARTY")]
    ),
    (
        "Since 2005, the Congress's foremost Jat leader in the state has steadily consolidated his hold over the community.",
        [("Congress", "PARTY")]
    ),
    (
        "In the past few years, the INLD has had to contend with the JJP, formed in 2018 by Abhay's nephew Dushyant Chautala.",
        [("INLD", "PARTY"), ("JJP", "PARTY"), ("Abhay", "POL_LEADER"), ("Dushyant Chautala", "POL_LEADER")]
    ),
    (
        "In immediate success, the JJP won 10 seats in the 2019 Haryana Assembly polls.",
        [("JJP", "PARTY")]
    ),
    (
        "The BJP and JJP contested separately.",
        [("BJP", "PARTY"), ("JJP", "PARTY")]
    ),
    (
        "The INLD, which had won just one seat in 2019, did slightly better in 2024.",
        [("INLD", "PARTY")]
    ),
    (
        "But it was way higher than the JJP's, which is what is boosting Abhay's hopes.",
        [("JJP", "PARTY"), ("Abhay", "POL_LEADER")]
    ),
    (
        "Abhay told The Indian Express: All our workers have returned to the INLD.",
        [("Abhay", "POL_LEADER"), ("INLD", "PARTY")]
    ),
    (
        "The JJP is finished.",
        [("JJP", "PARTY")]
    ),
    (
        "Haryana BJP president Mohan Lal Badoli said the public has rejected the INLD.",
        [("BJP", "PARTY"), ("Mohan Lal Badoli", "POL_LEADER"), ("INLD", "PARTY")]
    ),
    (
        "Congress media in-charge Chandvir Hooda claimed the INLD has no base left in Haryana.",
        [("Congress", "PARTY"), ("Chandvir Hooda", "POL_LEADER"), ("INLD", "PARTY")]
    ),
    (
        "Prime Minister Narendra Modi on Saturday is set to make his first visit to Manipur.",
        [("Narendra Modi", "POL_LEADER")]
    ),
    (
        "The Kuki-Zo welcomed the revised Suspension of Operations pact, while COCOMI condemned it.",
        [("Suspension of Operations", "SCHEME"), ("COCOMI", "PARTY")]
    ),
    (
        "Kuki-Zo groups demanded a National Register of Citizens (NRC) and the detection of illegal immigrants.",
        [("National Register of Citizens", "SCHEME"), ("NRC", "SCHEME")]
    ),
    (
        "The new terms of the SoO include relocation of camps from near Meitei areas.",
        [("SoO", "SCHEME")]
    ),
    (
        "Rahul Gandhi criticized the government's foreign policy.",
        [("Rahul Gandhi", "POL_LEADER")]
    ),
    (
        "Mamata Banerjee and Sonia Gandhi discussed opposition unity.",
        [("Mamata Banerjee", "POL_LEADER"), ("Sonia Gandhi", "POL_LEADER")]
    ),
    (
        "Arvind Kejriwal launched an anti-corruption helpline in Delhi.",
        [("Arvind Kejriwal", "POL_LEADER")]
    ),
    (
        "Akhilesh Yadav met students at a youth convention in Kanpur.",
        [("Akhilesh Yadav", "POL_LEADER")]
    ),
    (
        "PM Modi interacted with scientists at ISRO headquarters.",
        [("PM Modi", "POL_LEADER"), ("ISRO", "PARTY")]
    ),

    # ---------------- HINDI ----------------
    (
        "नरेंद्र मोदी ने दिल्ली में बीजेपी नेताओं से मुलाकात की।",
        [("नरेंद्र मोदी", "POL_LEADER"), ("बीजेपी", "PARTY")]
    ),
    (
        "राहुल गांधी ने कांग्रेस की रैली को संबोधित किया।",
        [("राहुल गांधी", "POL_LEADER"), ("कांग्रेस", "PARTY")]
    ),
    (
        "ममता बनर्जी ने कोलकाता में नई योजनाओं की घोषणा की।",
        [("ममता बनर्जी", "POL_LEADER")]
    ),
    (
        "अरविंद केजरीवाल ने दिल्ली में शिक्षा पहल शुरू की।",
        [("अरविंद केजरीवाल", "POL_LEADER")]
    ),
    (
        "शरद पवार ने एनसीपी की बैठक में भाग लिया।",
        [("शरद पवार", "POL_LEADER"), ("एनसीपी", "PARTY")]
    ),
    (
        "अमित शाह ने नई योजना शुरू की।",
        [("अमित शाह", "POL_LEADER")]
    ),

    # ---------------- TAMIL ----------------
    (
        "நரேந்திர மோடி டெல்லியில் பேசியார்.",
        [("நரேந்திர மோடி", "POL_LEADER")]
    ),
    (
        "ராகுல் காந்தி காங்கிரஸ் தலைவர்களை சந்தித்தார்.",
        [("ராகுல் காந்தி", "POL_LEADER"), ("காங்கிரஸ்", "PARTY")]
    ),
    (
        "மமதா பானர்ஜி கொல்கத்தாவில் புதிய திட்டங்களை அறிவித்தார்.",
        [("மமதா பானர்ஜி", "POL_LEADER")]
    ),
    (
        "மு.க. ஸ்டாலின் சென்னையில் அமைச்சரை சந்தித்தார்.",
        [("மு.க. ஸ்டாலின்", "POL_LEADER")]
    ),
    (
        "பாஜக மற்றும் காங்கிரஸ் இடையே விவாதம் நடந்தது.",
        [("பாஜக", "PARTY"), ("காங்கிரஸ்", "PARTY")]
    ),

    # ---------------- TELUGU ----------------
    (
        "నరేంద్ర మోదీ ఢిల్లీలో బీజేపీ కార్యకర్తలను ఉద్దేశించి మాట్లాడారు.",
        [("నరేంద్ర మోదీ", "POL_LEADER"), ("బీజేపీ", "PARTY")]
    ),
    (
        "రాహుల్ గాంధీ కాంగ్రెస్ నేతలను కలిశారు.",
        [("రాహుల్ గాంధీ", "POL_LEADER"), ("కాంగ్రెస్", "PARTY")]
    ),
    (
        "మమతా బెనర్జీ కోల్‌కతాలో కొత్త పథకాలు ప్రకటించారు.",
        [("మమతా బెనర్జీ", "POL_LEADER")]
    ),
    (
        "అర్వింద్ కేజ్రీవాల్ ఢిల్లీలో విద్యా కార్యక్రమం ప్రారంభించారు.",
        [("అర్వింద్ కేజ్రీవాల్", "POL_LEADER")]
    ),
    (
        "చంద్రబాబు నాయుడు విజయవాడలో డిజిటల్ కార్యక్రమాలు ప్రకటించారు.",
        [("చంద్రబాబు నాయుడు", "POL_LEADER")]
    ),
]


# =========================================================
# 3) Utility helpers
# =========================================================
def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def normalize_entities(entities: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [(clean_text(e), label) for e, label in entities]


def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# =========================================================
# 4) Prediction function
# =========================================================
def predict_entities(text: str) -> List[Tuple[str, str]]:
    """
    Returns entity spans as (entity_text, entity_label).
    Works with WordPiece tokenization and merges subwords properly.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)

    pred_ids = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

    entities = []
    current_entity = ""
    current_label = None

    for token, label_id in zip(tokens, pred_ids):
        if token in tokenizer.all_special_tokens:
            if current_entity:
                entities.append((clean_text(current_entity), current_label))
                current_entity = ""
                current_label = None
            continue

        label = id2label.get(int(label_id), "O")

        if label == "O":
            if current_entity:
                entities.append((clean_text(current_entity), current_label))
                current_entity = ""
                current_label = None
            continue

        if "-" not in label:
            continue

        prefix, tag = label.split("-", 1)
        piece = token[2:] if token.startswith("##") else token

        if prefix == "B":
            if current_entity:
                entities.append((clean_text(current_entity), current_label))
            current_entity = piece
            current_label = tag

        elif prefix == "I" and current_label == tag:
            if token.startswith("##"):
                current_entity += piece
            else:
                current_entity += " " + piece
        else:
            if current_entity:
                entities.append((clean_text(current_entity), current_label))
            current_entity = piece
            current_label = tag

    if current_entity:
        entities.append((clean_text(current_entity), current_label))

    return dedupe_preserve_order(entities)


# =========================================================
# 5) Description generator
# =========================================================
class PoliticalEntityDescriber:
    def __init__(self, request_delay: float = 0.3):
        self.cache: Dict[str, str] = {}
        self.request_delay = request_delay

    def _get_wikipedia_summary(self, entity: str) -> str:
        try:
            wikipedia.set_lang("en")
            try:
                return wikipedia.summary(entity, sentences=2, auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                try:
                    return wikipedia.summary(e.options[0], sentences=2, auto_suggest=False)
                except Exception:
                    return self._get_wikidata_description(entity)
            except wikipedia.PageError:
                return self._get_wikidata_description(entity)
        except Exception:
            return self._get_wikidata_description(entity)

    def _get_wikidata_description(self, entity: str) -> str:
        try:
            search_url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbsearchentities",
                "format": "json",
                "language": "en",
                "search": entity,
                "type": "item",
            }

            response = requests.get(search_url, params=params, timeout=10)
            data = response.json()

            if data.get("search"):
                entity_id = data["search"][0]["id"]
                detail_params = {
                    "action": "wbgetentities",
                    "format": "json",
                    "ids": entity_id,
                    "languages": "en",
                    "props": "descriptions|labels",
                }

                response = requests.get(search_url, params=detail_params, timeout=10)
                data = response.json()

                description = (
                    data["entities"][entity_id]
                    .get("descriptions", {})
                    .get("en", {})
                    .get("value")
                )
                if description:
                    return description

                label = (
                    data["entities"][entity_id]
                    .get("labels", {})
                    .get("en", {})
                    .get("value")
                )
                if label:
                    return label

        except Exception:
            pass

        return f"Information about {entity}"

    def describe_entity(self, entity: str) -> str:
        entity = clean_text(entity)
        if not entity:
            return "No entity provided"

        if entity in self.cache:
            return self.cache[entity]

        description = self._get_wikipedia_summary(entity)
        self.cache[entity] = description
        time.sleep(self.request_delay)
        return description


def add_descriptions_to_predictions(predict_entities_output: List[Tuple[str, str]]):
    """
    Takes output of predict_entities() and adds descriptions.
    Output:
      [
        {"entity": "...", "label": "...", "description": "..."},
        ...
      ]
    """
    describer = PoliticalEntityDescriber()

    entities = [entity for entity, _ in predict_entities_output]
    unique_entities = dedupe_preserve_order([clean_text(e) for e in entities])

    descriptions = {}
    for entity in unique_entities:
        descriptions[entity] = describer.describe_entity(entity)

    enriched_results = []
    for entity, label in predict_entities_output:
        entity = clean_text(entity)
        enriched_results.append({
            "entity": entity,
            "label": label,
            "description": descriptions.get(entity, "Information not found"),
        })

    return enriched_results


def print_enriched_results(enriched_results, original_sentence=""):
    if original_sentence:
        print(f"\nSentence: {original_sentence}")
    print("=" * 90)
    for result in enriched_results:
        print(f"{result['entity']} ({result['label']})")
        print(f"{result['description']}")
        print("-" * 90)


if __name__ == "__main__":
    # =========================================================
    # 6) Evaluation
    # =========================================================
    y_true, y_pred = [], []

    print("\n========== TESTING + DESCRIPTION OUTPUT ==========\n")

    for idx, (text, gold) in enumerate(test_data, start=1):
        gold = normalize_entities(gold)
        pred = normalize_entities(predict_entities(text))

        print(f"\n[{idx}] Text: {text}")
        print(f"Expected: {gold}")
        print(f"Predicted: {pred}")
        print("-" * 90)

        enriched_entities = add_descriptions_to_predictions(pred)
        print_enriched_results(enriched_entities, text)

        gold_set = set(gold)
        pred_set = set(pred)

        for e in gold_set.union(pred_set):
            y_true.append(1 if e in gold_set else 0)
            y_pred.append(1 if e in pred_set else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    accuracy = (
        sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        if y_true else 0.0
    )

    print("\n========== EVALUATION METRICS ==========")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")