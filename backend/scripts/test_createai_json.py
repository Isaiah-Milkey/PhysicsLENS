import os
import json
import requests
from pathlib import Path

api_key = os.getenv("CREATEAI_API_KEY")

if not api_key:
    raise RuntimeError("CREATEAI_API_KEY is not set.")

url = "https://api-main.aiml.asu.edu/query"

json_path = Path("test_reports/sample_diagnosis.json")

with json_path.open("r", encoding="utf-8") as f:
    report = json.load(f)

# Convert JSON into compact text.
report_text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))

prompt = f"""
You are a physics diagnostic assistant for AI-generated videos.

You will receive a structured JSON report from PhysicsLENS.
Your job is to write a concise final diagnosis for a human evaluator.

Important rules:
- Do not invent evidence that is not present in the JSON.
- If a score is described as prototype or first-pass, say that clearly.
- Distinguish between "no issue detected" and "no severity data reported".
- Focus on physics consistency, detected anomalies, severity, confidence, and recommended follow-up tests.

Please return the answer in this format:

# PhysicsLENS Final Diagnosis

## Overall Assessment
...

## Main Evidence
...

## Severity and Confidence
...

## Recommended Follow-up
...

Here is the PhysicsLENS JSON report:

{report_text}
""".strip()

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "query": prompt
}

response = requests.post(url, headers=headers, json=payload, timeout=90)

print("Status code:", response.status_code)

try:
    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception:
    print(response.text)