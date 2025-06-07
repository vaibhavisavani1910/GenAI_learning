from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
vector = embeddings.embed_documents(
    [
        "Delhi is the capital of India",
        "Kolkata is the capital of West Bengal",
        "Paris is the capital of France"
    ]
)
print(len(vector))

