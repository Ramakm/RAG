from dotenv import load_dotenv
load_dotenv()
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url = "https://api.groq.com/openai/v1"
)

# conversion = client.chat.completions.create( 
#     model = "openai/gpt-oss-120b",
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of France? and give us details about it."},
#     ],
# )
# print(conversion.choices[0].message.content)

# Groq  has no embedding models. So, embeddings use local Ollama

# First Download Ollama from https://ollama.com/download and install it. Then run the following command to start the Ollama server:
# ollama serve


embedding_client = OpenAI(
    api_key = "ollama",
    base_url = "http://localhost:11434/v1"
)

response = embedding_client.embeddings.create(
    model="nomic-embed-text",
    input="I am Ram who is teaching RAG to the students in the class. And Its been fun."
)

print(response)
print("We want to see the vector dimenssion size:")
print(len(response.data[0].embedding)) # 768


# Tokens = 1m token use cases
# Memory = 1GB Vector DB