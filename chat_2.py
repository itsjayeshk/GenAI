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

You are an AI assisstant who is expert in breaking down complex problems and then resolve the user query
For the given usr input analyze the user input and break it down into smaller steps and then solve the problem step by step and provide the final answer along with the explanation.

The steps are you get a suer input,you analyse,you think, you agai for several times and then you provide the final answer along with the explanation.

Follow this steps in sequence that is "analyse" -> "think" -> "output" -> "validate" -> "result"

Rules:
1. Follow the strict JSON output as per output schema
2. Always perform one step at a time and wait for the next input
3. Carefully analyse everything and user query

Output Format: 
{ step: "string", content: "string"}

Example: 
Input: What is 2 + 2.
Output: {{ step: "analyse", content: "Alright,the user is interested in maths and user is asking a basic arithematic operation}}
Output: {{ step: "think", content: "To perform the addition,i must go from left to right and perform the addition operation on the numbers provided in the user input
Output: {{ step: "output", content: "4"}}
Output: {{ step: "validate", content: "seems like 4 is the correct ans for 2 + 2"}}
Output: {{ step: "result" , content: "The final answer is 4 which is calculated by adding 2 with 2"}}
"""

result = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": "What is 2 + 2 * 0?"
        },
        {
            "role": "assistant",
            "content": json.dumps({"step": "analyse", "content": "Alright,the user is interested in maths and user is asking a basic arithematic operation"})
        },
        {
            "role": "assistant",
            "content": json.dumps({"step": "think",
   "content": "To solve this expression, I need to follow the order of operations (PEMDAS/BODMAS), which means I need to perform the multiplication operation first and then the addition operation"
})
        },
        {
            "role": "assistant",
            "content": json.dumps({
  "step": "output",
   "content": "2"
})
        }
    ]
)

print(result.choices[0].message.content)