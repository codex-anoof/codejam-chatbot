# Future Minds Ultimate RAG Chatbot (Multi-Agent Architecture)

import os
import json
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# === Agent 1: ReaderAgent ===
class ReaderAgent:
    def __init__(self, filepath):
        self.doc = fitz.open(filepath)

    def extract_pages(self):
        return [(f"Page {i+1}", page.get_text()) for i, page in enumerate(self.doc) if page.get_text().strip()]


# === Agent 2: SplitterAgent ===
class SplitterAgent:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, pages):
        docs = []
        for pid, text in pages:
            for chunk in self.splitter.split_text(text):
                docs.append(Document(page_content=chunk, metadata={"page": pid}))
        return docs


# === Agent 3: EmbedAgent ===
class EmbedAgent:
    def __init__(self):
        self.embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def build_vectorstore(self, docs):
        return FAISS.from_documents(docs, self.embedder)


# === Agent 4: RetrieverAgent ===
class RetrieverAgent:
    def __init__(self, vectorstore):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def retrieve(self, query):
        return self.retriever.get_relevant_documents(query)


# === Agent 5: AnswerAgent ===
class AnswerAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    def answer(self, query, retriever):
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
        return qa_chain.run(query)


# === Agent 6: ReferenceAgent ===
class ReferenceAgent:
    def extract_pages(self, docs):
        return ", ".join(sorted({doc.metadata["page"] for doc in docs}))


# === Orchestrator ===
class Orchestrator:
    def __init__(self, pdf_path, queries_path):
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDetjWLtpDucAF1ZfaN3xLBr-wC-sqVWKc"
        self.reader = ReaderAgent(pdf_path)
        self.splitter = SplitterAgent()
        self.embedder = EmbedAgent()
        self.ans_agent = AnswerAgent()
        self.ref_agent = ReferenceAgent()

        self.queries = json.load(open(queries_path))
        self.df_out = pd.DataFrame()

    def run(self):
        pages = self.reader.extract_pages()
        chunks = self.splitter.split(pages)
        vectorstore = self.embedder.build_vectorstore(chunks)
        retriever = RetrieverAgent(vectorstore)

        results = []
        for q in tqdm(self.queries):
            docs = retriever.retrieve(q["question"])
            context = "\n---\n".join([doc.page_content for doc in docs])
            answer = self.ans_agent.answer(q["question"], retriever.retriever)
            pages = self.ref_agent.extract_pages(docs)
            results.append({"ID": q["query_id"], "Context": context, "Answer": answer, "Sections": "N/A", "Pages": pages})

        self.df_out = pd.DataFrame(results)
        self.df_out.to_csv("lordofcode.csv", index=False)
        print("✅ submission.csv generated!")


# === Run if Main ===
if __name__ == "__main__":
    Orchestrator(
        pdf_path="data/grade-11-history-text-book.pdf",
        queries_path="data/queries.json"
    ).run()
