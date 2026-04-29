# Political Entity Detection

A multilingual political entity detection project built with a BERT-based token classification model. The system detects political leaders, political parties, and political schemes from text in multiple languages such as English, Hindi, Tamil, and Telugu. It also enriches detected entities with short descriptions using Wikipedia/Wikidata.

## Project Overview

This project focuses on identifying political entities from news articles, speeches, and mixed-language text. It uses a fine-tuned transformer model for named entity recognition and a small description generator to show additional information for each detected entity.

### Detected Entity Types

* `POL_LEADER` — Political leaders
* `PARTY` — Political parties or political organizations
* `SCHEME` — Government schemes, policies, or initiatives

## Features

* Multilingual political entity detection
* Supports English, Hindi, Tamil, and Telugu text
* Token classification using a BERT-based model
* Entity description generation using Wikipedia/Wikidata
* Streamlit frontend for easy testing
* Test script for model evaluation

## Project Structure

```text
Political_detection/
│
├── political-bert/                  # Fine-tuned model folder
│   ├── config.json
│   ├── model.safetensors
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── tokenizer.json
│
├── description_generator/
│   ├── __init__.py
│   └── generate_description.py     # Entity description utilities
│
├── testing/
│   └── test_model.py               # Model test and evaluation script
│
├── app.py                          # Streamlit frontend
└── README.md
```

## Requirements

Make sure you have Python installed. Then install the required packages:

```bash
pip install torch transformers streamlit wikipedia requests scikit-learn
```

If you face tokenizer issues, also install:

```bash
pip install sentencepiece
```

## How to Run the Project

### 1. Open Command Prompt

Go to your project folder:

```bash
cd /d C:\Users\shiva\OneDrive\Desktop\Political_detection
```

### 2. Activate the virtual environment

If you already created a virtual environment:

```bash
.venv\Scripts\activate
```

### 3. Run the test script

This checks model predictions and prints evaluation output:

```bash
python testing\test_model.py
```

### 4. Run the Streamlit frontend

This opens the interactive UI in your browser:

```bash
streamlit run app.py
```

## How It Works

1. The input text is passed through the trained BERT-based token classification model.
2. The model predicts entity labels for each token.
3. Predicted tokens are merged into full entity spans.
4. Detected entities are sent to the description generator.
5. Wikipedia/Wikidata summaries are displayed along with the entity labels.

## Sample Input

```text
PM Modi addressed a public rally in Varanasi.
```

## Sample Output

* `PM Modi` → `POL_LEADER`
* `Varanasi` → ignored if not labeled by the model
* Description of `PM Modi` is displayed below the result

## Notes

* Make sure the `political-bert` folder contains the trained model files.
* Keep `description_generator/__init__.py` empty so Python can import the module correctly.
* If you change the model files, rerun the test script before opening the frontend.

## Future Improvements

* Add more multilingual training data
* Improve entity span extraction accuracy
* Add backend API using FastAPI
* Build a React frontend
* Add downloadable JSON/CSV output

## Author

**Shivani Sankadiya**
