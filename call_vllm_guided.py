#!/usr/bin/env python3
# call_vllm_guided.py
# vLLM client with guided JSON extraction (Python 3.9 compatible)
# Make sure your vLLM server on Colab is running with the same model.

print("✅ Script started")

import requests
import json
import sys
import time

# vLLM server public URL (ngrok tunnel)
NGROK_BASE = "https://benmost-posthypnotic-darrick.ngrok-free.dev"
COMPLETIONS_URL = NGROK_BASE + "/v1/completions"
TIMEOUT = 90

def call_vllm_guided(prompt_dict, timeout=TIMEOUT):
    """Send the prompt (with guided_json) to vLLM and return parsed JSON output."""
    start = time.time()
    try:
        r = requests.post(COMPLETIONS_URL, json=prompt_dict, timeout=timeout)
        latency = time.time() - start
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("❌ Request failed:", e, file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print("Response text:", e.response.text, file=sys.stderr)
        raise

    print(f"HTTP {r.status_code}  latency: {latency:.2f}s")
    j = r.json()
    print("Raw JSON response (truncated):")
    print(json.dumps(j, indent=2, ensure_ascii=False)[:1500])

    text = j.get("choices", [])[0].get("text", "").strip()
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        print("⚠️ Could not parse JSON cleanly. Raw text below:")
        print(text)
        return text

if __name__ == "__main__":
    # Example note to extract TERM + ICD10 pairs
    prompt_text = (
        "System: You are a physician.\n"
        "User: Extract up to 3 (TERM, ICD10) diagnosis pairs from this note:\n"
        "'Patient presents with chest pain and shortness of breath; oxygen given.'\n"
        "Assistant: Return only valid JSON."
    )

    guided_schema = {
        "type": "array",
        "maxItems": 3,
        "items": {
            "type": "object",
            "properties": {
                "TERM": {"type": "string"},
                "ICD10": {"type": "string",
                          "pattern": "^[A-Z][0-9]{2}(\\.[0-9A-Z]{1,4})?$"}
            },
            "required": ["TERM", "ICD10"]
        }
    }

    payload = {
        "model": "andresnowak/Qwen3-0.6B-instruction-finetuned_v2",
        "prompt": prompt_text,
        "temperature": 0.0,
        "max_tokens": 200,
        "guided_json": guided_schema
    }

    try:
        result = call_vllm_guided(payload)
        print("\n=== Parsed guided JSON output ===\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        print("Request failed:", e, file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print("Response text:", e.response.text, file=sys.stderr)
