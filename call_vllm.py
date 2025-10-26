#!/usr/bin/env python3
# call_vllm_minimal.py
# Minimal client for vLLM /v1/completions (Python 3.9 compatible).

import requests
import sys
import json
import time

# <<< Replace with your active ngrok HTTPS base URL (no trailing slash) >>>
NGROK_BASE = "https://benmost-posthypnotic-darrick.ngrok-free.dev"
COMPLETIONS_URL = NGROK_BASE + "/v1/completions"

def build_prompt_from_messages(messages):
    system_parts = []
    user_parts = []
    for m in messages:
        role = m.get("role", "").lower()
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            # treat user / assistant / others as user text continuation
            user_parts.append(content)
    system_text = "\n\n".join(system_parts).strip()
    user_text = "\n\n".join(user_parts).strip()
    prompt = ""
    if system_text:
        prompt += "System: " + system_text + "\n\n"
    if user_text:
        prompt += "User: " + user_text + "\n\n"
    prompt += "Assistant:"
    return prompt

def call_vllm(prompt_dict, base_completions_url=COMPLETIONS_URL, timeout=60):
    # Build payload
    payload = {}

    if "model" in prompt_dict:
        payload["model"] = prompt_dict["model"]
    else:
        raise ValueError("prompt must include a 'model' key")

    if "messages" in prompt_dict:
        payload["prompt"] = build_prompt_from_messages(prompt_dict["messages"])
    elif "prompt" in prompt_dict:
        payload["prompt"] = prompt_dict["prompt"]
    else:
        raise ValueError("prompt must include either 'messages' or 'prompt'")

    # pass some optional generation params if given
    if "temperature" in prompt_dict:
        payload["temperature"] = prompt_dict["temperature"]
    if "max_tokens" in prompt_dict:
        payload["max_tokens"] = prompt_dict["max_tokens"]
    if "n" in prompt_dict:
        payload["n"] = prompt_dict["n"]

    try:
        start = time.time()
        r = requests.post(base_completions_url, json=payload, timeout=timeout)
        r.raise_for_status()
        latency = time.time() - start
    except requests.exceptions.RequestException as e:
        print("Request failed:", e, file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print("Response text:", e.response.text, file=sys.stderr)
        raise

    j = r.json()
    print("HTTP", r.status_code, "latency:", f"{latency:.2f}s")
    # print a short debug of returned JSON
    print(json.dumps(j, indent=2, ensure_ascii=False)[:2000])

    # Typical completions response shape: choices[0]["text"]
    try:
        return j.get("choices", [])[0].get("text")
    except Exception:
        return j

if __name__ == "__main__":
    # minimal prompt (mirror your original)
    prompt = {
        "model": "andresnowak/Qwen3-0.6B-instruction-finetuned_v2",
        "messages": [
            {"role": "system", "content": "You are a physician in the emergency department seeing a patient with a chief complaint."},
            {"role": "user", "content": "Can you tell me how to become rich?"}
        ],
        "temperature": 0.1,
        "max_tokens": 150
    }

    try:
        out = call_vllm(prompt)
        print("\n=== MODEL OUTPUT ===\n")
        if isinstance(out, str):
            print(out.strip())
        else:
            print(json.dumps(out, indent=2, ensure_ascii=False))
    except Exception as e:
        print("Error calling vLLM:", e, file=sys.stderr)
        sys.exit(1)
