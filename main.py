from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

# Sample Roman shipping data
text = """
Apollo carried 300 jars of wine on July 10.
Hermes arrived in Alexandria on August 3.
Zephyr departed from Corinth with olive oil on July 15.
Apollo docked in Rome on July 12.
Hermes transported textiles and spices from Carthage.
"""

chunks = [line.strip() for line in text.split('\n') if line.strip()]
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
embeddings = model.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def search(query):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), 1)
    return chunks[indices[0][0]]

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = None
    if request.method == "POST":
        query = request.form["query"]
        answer = search(query)
    return render_template("chat.html", answer=answer)

@app.route("/api", methods=["POST"])
def api():
    data = request.get_json()
    query = data.get("query", "")
    answer = search(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run()
