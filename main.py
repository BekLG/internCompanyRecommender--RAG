import os
from dotenv import load_dotenv
import json
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Load variables from the .env file
load_dotenv()

# Access variables
api_key = os.environ.get("API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key
)