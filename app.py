import streamlit as st
import os
import shutil
import fitz  # PyMuPDF
import docx
import re
import chromadb
from collections import Counter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === Paths ===
DATA_FOLDER = "data"
VECTOR_DB = "vector_store"
HISTORY_FILE = "history.txt"
os.makedirs(DATA_FOLDER, exist_ok=True)

# === Initialize Chroma DB ===
chroma_client = chromadb.PersistentClient(path=VECTOR_DB)
collection = chroma_client.get_or_create_collection(name="study_materials")

# === Load models ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Summarization model (manual to avoid meta tensor issues)
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cpu")
model.to(device)

# Session state for recent questions
if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = []

# === Helper functions ===
def run_summarization(text, max_len=180, min_len=50):
    """Summarize text using manual generation (safe on Windows CPU)."""
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_to_points(text, max_points=6):
    summary = run_summarization(text)
    # Split into bullet-like sentences
    points = [p.strip() for p in summary.replace("\n", " ").split(".") if p.strip()]
    return points[:max_points]

def extract_text_from_file(filepath):
    text = ""
    if filepath.endswith(".pdf"):
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
    elif filepath.endswith(".docx"):
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    return text

def chunk_text(text, chunk_size=500):
    text = re.sub(r'\s+', ' ', text)
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def clear_all_data():
    """Clear stored notes and vector DB."""
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)
        os.makedirs(DATA_FOLDER)
    if os.path.exists(VECTOR_DB):
        shutil.rmtree(VECTOR_DB)
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    global collection
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB)
    collection = chroma_client.get_or_create_collection(name="study_materials")

def highlight_terms(text, terms):
    """Highlight query terms inside a text."""
    for term in terms:
        if len(term) > 2:  # skip very small words
            text = re.sub(f"({term})", r"**\1**", text, flags=re.IGNORECASE)
    return text

def update_history(question, answer_points):
    """Append Q&A to history.txt."""
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"Q: {question}\n")
        for line in answer_points:
            f.write(f"- {line}\n")
        f.write("\n")

# === Streamlit UI ===
st.set_page_config(page_title="EduMate", layout="wide")
st.title("EduMate – Your AI Study Buddy")

tab1, tab2 = st.tabs(["📂 Upload Notes", "❓ Ask Questions"])

# ----------- TAB 1: UPLOAD ------------
with tab1:
    st.write("Upload your study materials (PDF, DOCX, TXT).")

    if st.button("Clear All Data"):
        clear_all_data()
        st.success("All uploaded notes and stored data have been cleared!")

    uploaded_files = st.file_uploader(
        "Upload study files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        all_chunks = []
        full_text = ""
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            text = extract_text_from_file(file_path)
            full_text += " " + text
            chunks = chunk_text(text, chunk_size=500)

            embeddings = embedder.encode(chunks).tolist()
            ids = [f"{uploaded_file.name}_{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, embeddings=embeddings, ids=ids)

            all_chunks.extend(chunks)

        st.success(f"Processed {len(uploaded_files)} file(s) with {len(all_chunks)} chunks.")

        # --- Stopword list for cleaner stats ---
        stopwords = {"a","an","and","the","to","of","in","on","for","with","is","are","was","were","this","that","by","be"}
        words = [w for w in re.findall(r'\w+', full_text.lower()) if w not in stopwords]

        # --- Stats ---
        word_count = len(words)
        freq = Counter(words)
        top_words = ", ".join([w for w, _ in freq.most_common(5)])
        st.info(f"Total Words: {word_count} | Top 5 keywords: {top_words}")

        # --- Word cloud ---
        if word_count > 0:
            wc = WordCloud(width=500, height=300, background_color="white").generate(" ".join(words))
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

# ----------- TAB 2: ASK QUESTIONS ------------
with tab2:
    st.subheader("Ask a question about your notes")

    # Show recent questions as buttons
    if st.session_state.recent_questions:
        st.write("Recent Questions:")
        cols = st.columns(len(st.session_state.recent_questions[-5:]))
        for i, q in enumerate(st.session_state.recent_questions[-5:]):
            if cols[i].button(q):
                st.session_state.selected_question = q
                st.experimental_rerun()

    query = st.text_input("Type your question:",
                          value=st.session_state.get("selected_question", ""))

    if query:
        # Retrieval
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=20  # use more chunks for better context
        )
        retrieved_text = " ".join(results['documents'][0])

        if len(retrieved_text) > 0:
            # Strict prompt: answer only from notes
            context_prompt = (
                f"Question: {query}\n\n"
                f"Context from notes: {retrieved_text}\n\n"
                "Answer based ONLY on the context above. Do not add extra knowledge."
            )
            points = summarize_to_points(context_prompt)

            # Show answer
            st.success("### AI Answer")
            for p in points:
                st.write(f"- {p}")

            # Save for history and recent questions
            st.session_state.recent_questions.append(query)
            update_history(query, points)

            # Download button
            answer_text = "\n".join(f"- {p}" for p in points)
            st.download_button(
                label="Download Answer as TXT",
                data=answer_text,
                file_name="edumate_answer.txt",
                mime="text/plain"
            )

            # Show related chunks with highlights
            search_terms = query.split()
            with st.expander("Show related text from your notes"):
                for i, doc in enumerate(results['documents'][0]):
                    st.markdown(highlight_terms(doc, search_terms))
