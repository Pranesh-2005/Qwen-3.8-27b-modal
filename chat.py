"""Local terminal chat client for the deployed Qwen3.8-27B endpoint.

Usage:
    python chat.py https://<workspace>--qwen38-27b-serve-serve.modal.run
"""
import sys

from openai import OpenAI

SERVED_NAME = "qwen3.8-27b"


def main():
    if len(sys.argv) < 2:
        print("Usage: python chat.py <modal-endpoint-url>")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/") + "/v1"
    client = OpenAI(base_url=base_url, api_key="not-needed")

    history = []
    print("Qwen3.8-27B chat. Ctrl+C to quit.")
    while True:
        try:
            user_msg = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not user_msg:
            continue

        history.append({"role": "user", "content": user_msg})
        print("qwen> ", end="", flush=True)

        stream = client.chat.completions.create(
            model=SERVED_NAME,
            messages=history,
            stream=True,
            max_tokens=2048,
        )
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            reply += delta
            print(delta, end="", flush=True)
        print()
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
