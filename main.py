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

def log_messages_to_list(filename, n):
  """
  Reads messages from a JSON file and stores the first n messages in a list named 'splits'.

  Args:
      filename: Path to the JSON file.
      n: Number of messages to store.
  """

  with open(filename, "r") as f:
      data = json.load(f)

  # Ensure data is a list of messages
  if not isinstance(data, list):
      raise ValueError("JSON data is not a list of messages.")

  splits = []  # Initialize an empty list to store messages

  for message in data[:n]:
      if len(splits) >= n:  # Limit to n messages
          break

      try:
          message_content = message["message"]
          splits.append(message_content)  # Add message content to the list
      except KeyError:
          print(f"Message does not contain a 'message' key.")

  return splits  # Return the list of messages

# Sample JSON file path (replace with your actual path)
json_file = "/home/bek/Downloads/SQLite.json_modified.json"

# Number of messages to store
num_messages_to_store = 1000

# Get messages and store them in splits
texts = log_messages_to_list(json_file, num_messages_to_store)

print(f"Stored the first {num_messages_to_store} messages in 'splits'.")



persist_directory = 'docs/chroma/'

# Create the vector store
vector_database = Chroma.from_texts(texts, embeddings)