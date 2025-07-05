import os
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model

load_dotenv()


model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
res = model.invoke("Hello, world!")

print(res.content)