# 🤖 Future Minds RAG Chatbot

A multi-agent GenAI chatbot built in **Python** for the **CodeJam Future Minds** competition.  
It uses **Gemini 1.5 Flash** for Retrieval-Augmented Generation (RAG) to answer questions from a **Grade 11 History** textbook. 📚✨

---

## 🚀 Features

- 🤖 Powered by **Google Generative AI (Gemini)**
- 📁 Integrated with **FAISS Vector DB** for efficient retrieval
- 📌 Extracts references (📄 *Sections* & 📑 *Pages*) from textbook
- ⚡ Supports **Batch Query Evaluation** for multiple questions

---

## 🛠️ Setup

1. 📦 Create a virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. 📚 Place your textbook PDF inside the `data/` folder.

3. 🧠 Run the chatbot:

   ```bash
   python src/rag_bot.py
   ```

---

## 🔑 Gemini API Key

Set your Gemini API key directly in `rag_bot.py` as:

```python
GOOGLE_API_KEY = "your-api-key-here"
```

---

## 📄 Output

- Generates a **`lordofcode.csv`** in the root directory with detailed results. 📊

---

