# liquid3_ollama_bot.py

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import subprocess

# STEP 1: Load and Split Your Knowledge Base
loader = TextLoader("liquid3_knowledge_base.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# STEP 2: Generate Embeddings and Create Chroma Vector DB
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding, persist_directory="./liquid3_db")
db.persist()

# STEP 3: Function to Answer Questions
def ask_liquid3_bot(question):
    retrieved_docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Prompt for Ollama
    prompt = f"""Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"""

    # Call Ollama
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        capture_output=True
    )

    print("\n🤖 Answer:")
    print(result.stdout.decode())

# STEP 4: User Input Loop
while True:
    q = input("\n💬 Ask Liquid 3 Bot (or type 'exit'): ")
    if q.lower() == "exit":
        break
    ask_liquid3_bot(q)
