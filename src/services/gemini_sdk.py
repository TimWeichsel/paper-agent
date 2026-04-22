from google import genai
import time
import re

_client = genai.Client() # The client gets the API key from the environment variable `GEMINI_API_KEY`.

def ask_gemini(prompt: str, model = "gemini-3-flash-preview") -> str:
    answer = _client.models.generate_content(
        model=model, contents=prompt
    )
    answer = answer.text.strip().strip("*").strip()
    return answer

def ask_gemini_with_retries(prompt: str, model = "gemini-3-flash-preview", retries=10, delay=10) -> str:
    print(f"Asking Gemini with prompt: {prompt}")
    for attempt in range(retries):
        try:
            return ask_gemini(prompt, model)
        except Exception as e:
            if "PerDay" in str(e):
                raise Exception("Daily quota exceeded. Try again tomorrow.") from e
            match = re.search(r'retry in (\d+)', str(e))
            wait = int(match.group(1)) + 1 if match else delay
            print(f"Attempt {attempt + 1} failed. Retrying in {wait}s...")
            time.sleep(wait)
    raise Exception("All attempts failed")