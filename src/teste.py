import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY não encontrado no .env")

genai.configure(api_key=api_key)

models = list(genai.list_models())
print(f"Modelos encontrados: {len(models)}\n")

for model in models:
    methods = getattr(model, "supported_generation_methods", None)
    if methods:
        print(f"{model.name}: {methods}")

print("\nModelos com suporte a generateContent / chat:")
for model in models:
    methods = getattr(model, "supported_generation_methods", None)
    if methods and any(m in methods for m in ["generateContent", "chat", "generate"]):
        print(f"- {model.name}: {methods}")
