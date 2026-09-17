from openai import OpenAI
import json

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = client.models.list().data[0].id

messages = [{"role": "user", "content": "Je 9.11 väčšie ako 9.8? Vysvetli."}]

resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=2000)
print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))