from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
import json
import re

client = OpenAI()

# Define the data model
class UserProfile(BaseModel):
    name: str
    age: int
    occupation: str

# Function to call the LLM
def generate_user_profile(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content

# Function to extract JSON from potentially noisy output
def extract_json(text: str) -> str:
    # Try to find JSON object in the text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text

# Retry mechanism with tenacity
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_validated_user_profile(prompt: str) -> UserProfile:
    output = generate_user_profile(prompt)
    try:
        # First try direct parsing
        user_profile = UserProfile.model_validate_json(output)
        return user_profile
    except ValidationError as e:
        # Try to extract JSON from the output
        try:
            cleaned_output = extract_json(output)
            user_profile = UserProfile.model_validate_json(cleaned_output)
            return user_profile
        except ValidationError:
            print(f"Validation error: {e}")
            print(f"LLM output was: {output}")
            raise

# Example usage
prompt = """Generate a JSON object with the user's name, age, and occupation.
Return ONLY valid JSON format with no additional text. Example:
{
  "name": "John Doe",
  "age": 30,
  "occupation": "Software Engineer"
}"""

try:
    user_profile = get_validated_user_profile(prompt)
    print("Successfully generated valid user profile:")
    print(user_profile.model_dump_json(indent=2))
except Exception as e:
    print(f"Failed to generate a valid user profile after multiple attempts: {e}")