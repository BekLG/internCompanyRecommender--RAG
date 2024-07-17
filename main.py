import os
from dotenv import load_dotenv
import json
import re
from flask import Flask, request ,jsonify
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

app = Flask(__name__)

# Load variables from the .env file
load_dotenv()

# Access variables
api_key = os.environ.get("API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key
)

persist_directory = 'docs/chroma/'

# Function to clean json responses which are from the LLM
def clean_json_string(json_string):
    # Remove unwanted escape sequences
    cleaned_string = re.sub(r'\\n|\\', '', json_string)
    return cleaned_string

# Function to be called when the server starts
def startup_function():
    print("Server is starting...")
    json_file = "SQLite.json_modified.json"
    num_messages_to_store = 200
    texts = log_messages_to_list(json_file, num_messages_to_store)
    global vector_database
    vector_database = Chroma.from_texts(texts, embeddings)

# Route handler for /ask
@app.route('/ask', methods=['POST'])
def ask_handler():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question') if data else None
        print(question)
        if question:
            # Similarity search with k = 5
            docs = vector_database.similarity_search(question, k=5)
            all_job_postings = ""
            for i in range(5):
                job_posting = f"Job {i+1}:\n"
                job_posting += f"  Text: {docs[i]}\n"
                job_posting += "-------------------\n"
                all_job_postings += job_posting

            # Generate response using generative model
            response = model.generate_content("""Input:

User's skills and expertise: {question}
                                              
A list containing the text descriptions of 5 job postings.

{""" +  all_job_postings +"""}


Instructions:
1. Analyze the provided job postings to understand the skills and requirements mentioned.
2. Generate recommendations for internship opportunities based on relevance to the user's skills and expertise.
3. For each recommendation, include:
   - Company Name: Name of the company offering the internship.
   - Internship Title: Title of the internship position.
   - Company Location: Location of the company (city, state, country, etc.).
   - Description: Explain why this internship is a good match for the user's skills and expertise.
4. Ensure the recommendations are ordered by relevance, with the most relevant opportunities listed first.

Output:
A JSON object with the following structure:
{
  "recommendations": [
    {
      "company": "Company Name",
      "title": "Internship Title",
      "location": "Company Location",
      "description": "Why this internship is a good match for the user's skills and expertise."
    }
  ]
}


""")


           
            print(response.text)
            cleaned_response = clean_json_string(response.text)
            return jsonify(cleaned_response)
        else:
            return 'No question provided'
    else:
        return 'Invalid request method'

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

if __name__ == '__main__':
    # Call the startup function when the server starts
    startup_function()

    # Run the Flask app
    app.run(debug=True)
