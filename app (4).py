import gradio as gr
from pypdf import PdfReader
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chunks = []
vectorizer = None
chunk_vectors = None

def upload_pdf(file):
    global chunks, vectorizer, chunk_vectors
    reader = PdfReader(file.name)
    text = "".join([page.extract_text() for page in reader.pages])
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    return f"✅ Uploaded! {len(chunks)} chunks created. You can ask questions now."

def ask_question(question):
    if chunk_vectors is None:
        return "Please upload a PDF first."

    q_vector = vectorizer.transform([question])
    scores = cosine_similarity(q_vector, chunk_vectors)[0]
    top_indices = scores.argsort()[::-1][:3]
    context = "\n\n".join([chunks[i] for i in top_indices])

    prompt = f"""Answer using only this context. If not found, say you don't know.

Context:
{context}

Question: {question}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

with gr.Blocks(title="My NotebookLM") as demo:
    gr.Markdown("# 📘 My NotebookLM")

    with gr.Row():
        pdf_input = gr.File(label="Upload your PDF")
        upload_btn = gr.Button("Upload")
    upload_status = gr.Textbox(label="Status")

    question_input = gr.Textbox(label="Ask a question about your PDF")
    ask_btn = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer")

    upload_btn.click(upload_pdf, inputs=pdf_input, outputs=upload_status)
    ask_btn.click(ask_question, inputs=question_input, outputs=answer_output)

demo.launch()
