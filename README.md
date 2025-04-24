# Future Minds RAG Chatbot

This is a multi-agent chatbot built in Python for the CodeJam Future Minds competition.
It uses Gemini 1.5 Flash for Retrieval-Augmented Generation (RAG) to answer questions from a Grade 11 History textbook.

## Features
- Google Generative AI (Gemini)
- FAISS Vector DB
- Reference Extraction (Sections and Pages)
- Batch query evaluation

## Setup

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the textbook PDF in `data/` folder.

3. Run the chatbot:

```bash
python src/rag_bot.py
```

## Gemini API Key
Your key is set directly in the `rag_bot.py` file as `GOOGLE_API_KEY`.

## Output
- Generates a `submission.csv` in the root directory.
