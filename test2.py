from openai import OpenAI
from transformers import AutoTokenizer
import requests

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
MODEL = client.models.list().data[0].id

messages = [{"role": "user", "content": "Je 9.11 väčšie ako 9.8? Vysvetli."}]

resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=2000)

tok = AutoTokenizer.from_pretrained(MODEL)

naive = len(tok(messages[0]["content"]).input_ids)

prompt_str = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
templated = len(tok(prompt_str).input_ids)

server = requests.post(
    "http://localhost:8000/tokenize",
    json={"model": MODEL, "messages": messages, "add_generation_prompt": True},
).json()["count"]

print(f"naive     = {naive}")
print(f"template  = {templated}")
print(f"server    = {server}")
print(f"usage     = {resp.usage.prompt_tokens}")
print()
print("--- čo model naozaj dostal na vstupe ---")
print(repr(prompt_str))