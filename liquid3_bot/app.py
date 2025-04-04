from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load AI Model
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def home():
    return {"message": "AI Model is Running!"}

@app.get("/embed")
def get_embedding(text: str):
    embedding = model.encode(text).tolist()
    return {"embedding": embedding}

# Run API (For local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
