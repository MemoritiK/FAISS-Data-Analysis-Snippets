import requests
import json
import os
API_KEY = os.getenv("API_KEY")
def get_explanation(content):   
    response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
     },
      data=json.dumps({
        "model": "kwaipilot/kat-coder-pro:free",
        "messages": [
          {
            "role": "user",
            "content": f"Explain the given code:\n {content}"
          }
        ]
      })
    )
    
    resp_json = response.json()
    
    answer = resp_json["choices"][0]["message"]["content"]
    return (answer)