
# to be verified

import OpenAI from "openai";
import os
from dotenv import load_dotenv
load_dotenv()

const openai = new OpenAI({
  apiKey: os.getenv("OPENAI_API_KEY"),
  baseURL: `https://oai.helicone.ai/v1`,
# To create a user go to http://localhost:54323/project/default/auth/users and add your account. You can use this account to sign into Helicone at localhost:3000 via your browser.
  # baseURL: `https://localhost:8787/v1/gateway/oai/v1`,
  defaultHeaders: {
    "Helicone-Auth": `Bearer ${process.env.HELICONE_API_KEY}`,
  },
});

response = openai_client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Count to 5",
    stream=False,
)
assert response.choices[0].text is not None
