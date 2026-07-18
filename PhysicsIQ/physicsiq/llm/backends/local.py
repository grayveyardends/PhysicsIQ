"""local.py — the local gguf backend: HTTP to llama-server on localhost.

llama-server exposes an OpenAI-compatible /v1/chat/completions endpoint, so
our internal message shape (see backends/base.py) goes over the wire as-is.
This is the backend the hackathon demo runs on: everything on the 4060,
nothing leaves the machine.

The process that serves this is owned by llm/server.py.
"""

import json

import requests

from physicsiq.llm.backends.base import GenerationCancelled, split_thinking


class LlamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def chat_stream(self, messages, temperature=0.35, max_tokens=2500,
                    grammar: str = None, on_chunk=None, on_thinking=None,
                    should_cancel=None) -> str:
        """Stream one completion. Returns the full ANSWER text when done.

        on_chunk(str)     — called with each new answer fragment
        on_thinking(str)  — reasoning models emit a chain-of-thought first;
                            llama-server splits it into `reasoning_content`
                            deltas when it was started with --jinja. We
                            surface it separately so the UI can show a dim
                            "thinking…" trace, and we NEVER put it in the
                            returned text (it would confuse code parsing).
        should_cancel()   — polled between chunks; return True to abort
        grammar           — optional GBNF grammar string. llama.cpp then
                            REFUSES to sample tokens that break the grammar.
                            This is our "logit checker": output is forced to
                            be structurally valid before it exists.

        max_tokens is generous by default because thinking, when it happens,
        spends from the same budget — too small a cap means the model thinks
        forever and answers nothing.
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
                thought = delta.get("reasoning_content") or ""
                if thought and on_thinking is not None:
                    on_thinking(thought)
                piece = delta.get("content") or ""
                if piece:
                    full.append(piece)
                    if on_chunk is not None:
                        on_chunk(piece)

        # Second line of defence: if the server did NOT split the thinking out
        # for us, do it here. The UI streamed the raw text (nice — you can
        # watch it think), but the ANSWER we return, and therefore the history
        # and the code the agent runs, is thought-free.
        answer, _thinking = split_thinking("".join(full))
        return answer

    def chat(self, messages, **kw) -> str:
        """Non-streaming convenience for short internal calls
        (e.g. 'does this screenshot match the request? yes/no')."""
        return self.chat_stream(messages, **kw)
