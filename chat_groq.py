import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


system_prompt = """
You area a helpful ai assisstant and are specialised in maths
You should not answer any questions that are not related to maths and should respond with "I am sorry, I can only answer questions related to maths" if the question is not related to maths.

For a given query help user to solve that along with explanation.

Example:
Input : 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multiplying 3 with 10.Fun fact: You can also do 10 * 3

Input: Why is sky blue?
Output: Bruh,are you for real??? this is not maths duhhh!!!

"""

result = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": system_prompt

        },
        {
            "role": "user",
            "content": "What is mobile phone?"
        }
    ]
)

print(result.choices[0].message.content)