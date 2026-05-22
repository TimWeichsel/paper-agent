
import time
import os
from groq import Groq


client = Groq(api_key=os.environ.get("GROQ_API_KEY")) # The client gets the API key from the environment variable `GROQ_API_KEY`.

def ask_groq(prompt: str, model = "llama-3.3-70b-versatile") -> str:
    answer = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt
        }],
        model=model,
    )
    answer = answer.choices[0].message.content.strip().strip("*").strip()
    return answer

def ask_groq_with_retries(prompt: str, model = "llama-3.3-70b-versatile", retries=10, delay=10) -> str:
    for attempt in range(retries):
        try:
            return ask_groq(prompt, model)
        except Exception as e:
            if "429" in str(e):
                raise Exception("Token quota exceeded. Try again tomorrow.") from e
            print(f"Attempt {attempt + 1} failed. Reason: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    raise Exception("All attempts failed ")