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
json_file = "SQLite.json_modified.json"

# Number of messages to store
num_messages_to_store = 1000

# Get messages and store them in splits
texts = log_messages_to_list(json_file, num_messages_to_store)

print(f"Stored the first {num_messages_to_store} messages in 'splits'.")



persist_directory = 'docs/chroma/'

# Create the vector store
vector_database = Chroma.from_texts(texts, embeddings)

vector_index = vector_database.as_retriever(search_kwargs={"k": 5})


question = ".net developer"

# Similarity search with k = 5
docs = vector_database.similarity_search(question,k=5)

# Create an empty string to store the job postings
all_job_postings = ""

# Iterate through the job postings
for i in range(5):
    job_posting = f"Job {i+1}:\n"
    job_posting += f"  Text: {docs[i]}\n"
    job_posting += "-------------------\n"
    
    # Append the current job posting to the variable containing all job postings
    all_job_postings += job_posting

# Print all job postings at once
# print(all_job_postings)

response = model.generate_content("""Input:

A list containing the text descriptions of 5 job postings.

{""" +  all_job_postings +"""}

Output:

A JSON object containing information extracted from each job posting.

Each entry in the JSON object should correspond to a single job posting.

For each job posting, include the following fields:

job_title: The title of the job (if available, otherwise "unknown").

company_name: The name of the company offering the job (if available, otherwise "unknown").

location: The location of the company (city, state, country, etc., if available, otherwise "unknown", if it is "Anywhere/remote" also mark it as "unknown" ).""")
print(response.text)