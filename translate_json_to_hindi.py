import json
import argparse
from deep_translator import GoogleTranslator

def translate_text(text):
    try:
        return GoogleTranslator(source="auto", target="hi").translate(text)
    except Exception:
        return text
def translate_text(text):
    try:
        return translator.translate(text, dest="hi").text
    except Exception:
        return text

def translate_obj(obj):
    if isinstance(obj, dict):
        return {k: translate_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [translate_obj(i) for i in obj]
    elif isinstance(obj, str):
        return translate_text(obj)
    else:
        return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_file", required=True)
    args = parser.parse_args()

    input_path = args.input_file

    if input_path.endswith(".jsonl"):
        output_path = input_path.replace(".jsonl", "_hi.jsonl")
        with open(input_path, "r", encoding="utf-8") as f, \
             open(output_path, "w", encoding="utf-8") as out:
            for line in f:
                data = json.loads(line)
                translated = translate_obj(data)
                out.write(json.dumps(translated, ensure_ascii=False) + "\n")
    else:
        output_path = input_path.replace(".json", "_hi.json")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        translated = translate_obj(data)
        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(translated, out, ensure_ascii=False, indent=2)

    print("Translation completed.")
    print("Saved as:", output_path)

if __name__ == "__main__":
    main()