"""client.py — the ONLY file that speaks HTTP to the model.

llama-server exposes an OpenAI-compatible /v1/chat/completions endpoint,
so `messages` below is the standard OpenAI shape:

    [{"role": "system", "content": "..."},
     {"role": "user",   "content": "..."},
     {"role": "user",   "content": [           # multimodal turn
         {"type": "text", "text": "what is this?"},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]

Design rule: NO Qt in this file. It takes a plain-python `on_chunk`
callback, which makes it testable from a terminal. The Qt marshalling
(callback → signal → GUI) happens in ui/workers.py.
"""

import base64
import json

import requests


class GenerationCancelled(Exception):
    """Raised inside the stream loop when the user hits Stop."""


def image_part(png_bytes: bytes) -> dict:
    """Wrap raw PNG bytes as an OpenAI image content-part.

    llama-server accepts base64 data URIs when started with --mmproj,
    which is how we show the model screenshots and PDF pages."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}}


class LlamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat_stream(self, messages, temperature=0.35, max_tokens=1400,
                    grammar: str = None, on_chunk=None,
                    should_cancel=None) -> str:
        """Stream one completion. Returns the full text when done.

        on_chunk(str)     — called with each new text fragment
        should_cancel()   — polled between chunks; return True to abort
        grammar           — optional GBNF grammar string. llama.cpp then
                            REFUSES to sample tokens that break the grammar.
                            This is our "logit checker": output is forced
                            to be structurally valid before it exists.
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if grammar:
            payload["grammar"] = grammar

        full = []
        # stream=True + iter_lines = we get Server-Sent Events line by line:
        #   data: {"choices":[{"delta":{"content":"Hel"}}]}
        #   data: {"choices":[{"delta":{"content":"lo"}}]}
        #   data: [DONE]
        with requests.post(self.base_url + "/v1/chat/completions",
                           json=payload, stream=True, timeout=(10, 600)) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if should_cancel is not None and should_cancel():
                    raise GenerationCancelled()
                if not raw:
                    continue  # SSE keep-alive blank lines
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue  # malformed event — skip, don't crash the chat
                piece = delta.get("content") or ""
                if piece:
                    full.append(piece)
                    if on_chunk is not None:
                        on_chunk(piece)
        return "".join(full)

    def chat(self, messages, **kw) -> str:
        """Non-streaming convenience for short internal calls
        (e.g. 'does this screenshot match the request? yes/no')."""
        return self.chat_stream(messages, **kw)
