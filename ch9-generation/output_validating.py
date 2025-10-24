# to be verified

from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
import os


client = OpenAI()
# Define the data model
class UserProfile(BaseModel):
    name: str
    age: int
    occupation: str

# Function to call the LLM
def generate_user_profile(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].message.content

# Retry mechanism with tenacity
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_validated_user_profile(prompt: str) -> UserProfile:
    output = generate_user_profile(prompt)
    try:
        # Attempt to parse and validate the LLM output
        user_profile = UserProfile.parse_raw(output)
        return user_profile
    except ValidationError as e:
        print(f"Validation error: {e}")
        raise

# Example usage
prompt = "Generate a JSON object with the user's name, age, and occupation."
try:
    user_profile = get_validated_user_profile(prompt)
    print(user_profile)
except Exception as e:
    print(f"Failed to generate a valid user profile after multiple attempts: {e}")
