from dotenv import load_dotenv
from google import genai
import arxiv

load_dotenv() # Load environment variables from .env file
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

contents= "Give me a joke about copenhagen"

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents=contents
)
print(response.text)