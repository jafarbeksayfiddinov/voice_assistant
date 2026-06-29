import data

import json

import warnings

warnings.filterwarnings("ignore")

import requests


def ask_ollama(question, context):
    messages = [
        {
            "role": "system",
            "content": data.DOC_SYSTEM.format(context=context)
        },
        {
            "role": "user",
            "content": question
        }
    ]

    payload = {
        "model": data.MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.0,
            "num_predict": 80,
            "repeat_penalty": 1.3
        },
    }

    full = ""

    with requests.post(
        data.OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=120
    ) as r:

        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)

                token = chunk.get("message", {}).get("content", "")

                print(token, end="", flush=True)

                full += token

                if chunk.get("done"):
                    break

    print()

    return full.strip()