import json
import regex as re
from typing import Any, Tuple
from dirtyjson import  loads as dirty_loads
from pydantic import BaseModel, ValidationError, Field

# ------------ 1.  Schema ------------
class UserProfile(BaseModel):
    id: int
    name: str
    score: float = Field(..., ge=0)

# ------------ 2.  Helpers ------------
BRACES_RE = re.compile(r"\{(?:[^{}]+|(?R))*\}", re.DOTALL)

def extract_json_block(text: str) -> str:
    """Return the outer-most {...} block or raise."""
    match = BRACES_RE.search(text)
    if not match:
        raise ValueError("No JSON object found")
    return match.group(0)

def tolerant_parse(raw: str) -> Any:
    """Parse raw JSON or 'JSON-ish' text into Python objects."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return dirty_loads(raw)  # handles quotes, commas, comments
        except ValueError as e:
            raise ValueError("Unrecoverable JSON") from e

def validate(obj: Any) -> UserProfile:
    """Validate & coerce parsed data into a strongly-typed object."""
    try:
        return UserProfile.model_validate(obj)
    except ValidationError as e:
        # Normalize field names (e.g., "ID" -> "id")
        normalized = {k.lower(): v for k, v in obj.items()}
        try:
            return UserProfile.model_validate(normalized)
        except Exception:
            raise
# ------------ 3.  Pipeline ------------
def clean_llm_output(text: str) -> Tuple[UserProfile, str]:
    json_block = extract_json_block(text)
    parsed = tolerant_parse(json_block)
    validated = validate(parsed)
    normalised = json.dumps(validated.model_dump(), separators=(",", ":"), sort_keys=True)
    return validated, normalised

# ------------ 4.  Demo ------------
if __name__ == "__main__":
    dirty_reply = """
    Sure! Here is the JSON:
    { "ID": 123 , "Name": "Alice", "score":  95, }
    Hope it helps!
    """

    dirty_reply = """
    Sure! Here is the JSON:
    { "messages":"this is a test ", "ID": 123 , "Name": "Alice", "score":  95, } hello world
    Hope it helps!
    """
    dirty_reply = """
    Sure! Here is the JSON:
    { "messages":"this is a test ", "ID": 123 , "Name": "Alice", "score":  95,  hello world
    Hope it helps!
    """

    profile, final_json = clean_llm_output(dirty_reply)
    print(profile)       # typed Pydantic object
    print(final_json)    # {"id":123,"name":"Alice","score":95}
