from dotenv import load_dotenv
import os
import json
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

system_prompt = """

You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input analyze the user input and break it down into smaller steps and then solve the problem step by step and provide the final answer along with the explanation.

Follow this steps in sequence that is "analyse" -> "think" -> "output" -> "validate" -> "result"

Rules:

1. Follow the strict JSON output as per output schema
2. Always perform one step at a time and wait for the next input
3. Carefully analyse everything and user query
4. Return ONLY valid JSON

Output Format:
{"step": "string", "content": "string"}

Example:
Input: What is 2 + 2.

Output:
{"step": "analyse", "content": "Alright, the user is interested in maths and is asking a basic arithmetic operation"}

Output:
{"step": "think", "content": "To perform the addition, I need to add 2 and 2"}

Output:
{"step": "output", "content": "4"}

Output:
{"step": "validate", "content": "2 + 2 equals 4"}

Output:
{"step": "result", "content": "The final answer is 4"}
"""

messages = [
    {"role": "system", "content": system_prompt},
]

query = input("> ")
messages.append({"role": "user", "content": query})

while True:

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
        messages=messages
    )

    parsed_response = json.loads(
        response.choices[0].message.content
    )

    messages.append({
        "role": "assistant",
        "content": json.dumps(parsed_response)
    })

    if parsed_response["step"] == "result":
        print("🤖: " + parsed_response.get("content"))
        break

    print("🧠: " + parsed_response.get("content"))