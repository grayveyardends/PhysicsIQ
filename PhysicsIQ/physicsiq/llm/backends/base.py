"""base.py — what every LLM backend must provide, plus the bits they share.

THE INTERFACE (that's all of it):

    backend.chat_stream(messages, temperature=…, max_tokens=…, grammar=None,
                        on_chunk=None, on_thinking=None,
                        should_cancel=None) -> str   # the ANSWER text

`messages` is always the OpenAI shape, whatever the backend is underneath:

    [{"role": "system", "content": "..."},
     {"role": "user",   "content": "..."},
     {"role": "user",   "content": [                 # multimodal turn
         {"type": "text", "text": "what is this?"},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}]}]

Backends translate that shape into whatever their wire format is. Keeping ONE
internal shape is why `ui/workers.LlmStreamWorker` and `agent/loop.py` don't
know or care which model is answering.

Rules for a backend:
  * NO Qt in here. Backends take plain-python callbacks; the Qt marshalling
    (callback -> signal -> GUI) happens in ui/workers.py. That's what makes
    these testable from a terminal.
  * on_chunk / on_thinking are called from a WORKER thread. Emit, don't touch.
  * should_cancel() is polled between chunks; raise GenerationCancelled.
"""

import base64


class GenerationCancelled(Exception):
    """Raised inside a stream loop when the user hits Stop."""


def image_part(image_bytes: bytes, mime: str = "image/png") -> dict:
    """Wrap raw image bytes as an OpenAI image content-part.

    mime: "image/png" or "image/jpeg" — must match the actual bytes."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}}


def load_image_file(path: str) -> dict:
    """Read an image FILE from disk into a content-part, picking the mime
    type from the extension. Used by the chat panel's attach button."""
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    with open(path, "rb") as fh:
        return image_part(fh.read(), mime)


def decode_image_part(part: dict):
    """The reverse: an OpenAI image part -> (mime, raw base64 string).
    Backends that don't speak data-URIs (Anthropic wants the mime and the
    payload in separate fields) use this."""
    url = part["image_url"]["url"]
    header, _, payload = url.partition(",")
    mime = header[len("data:"):].split(";", 1)[0] or "image/png"
    return mime, payload


def split_thinking(text: str):
    """(answer, thinking) — pull a reasoning model's chain-of-thought out of
    the answer text.

    Reasoning models (DeepSeek-R1 and friends) wrap their musings in literal
    <think>…</think> tags. llama-server only splits those into a separate
    `reasoning_content` field when it was started with --jinja AND recognizes
    the template; otherwise the whole thought lands in `content`. If we let
    that through, the agent parses a code fence out of the model's
    HALF-FINISHED reasoning and runs it. So we always cut it ourselves.

    Handles the tagless-open case too: llama.cpp pre-fills "<think>" into the
    prompt for R1, so the model often emits only the CLOSING tag.
    """
    if "</think>" in text:
        thinking, _, answer = text.rpartition("</think>")
        return answer.strip(), thinking.replace("<think>", "").strip()
    if "<think>" in text:      # opened but never closed = ran out of budget
        head, _, thought = text.partition("<think>")
        return head.strip(), thought.strip()
    return text, ""


def flatten_messages(messages):
    """(system_text, turns) where turns is [(role, text, [image_parts]), …].

    Backends that take a plain prompt instead of a message array (the
    `claude` CLI) use this to render the conversation as a transcript.
    """
    system_bits, turns = [], []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            text, images = content, []
        else:
            text = "\n".join(p["text"] for p in content
                             if p.get("type") == "text")
            images = [p for p in content if p.get("type") == "image_url"]
        if msg["role"] == "system":
            system_bits.append(text)
        else:
            turns.append((msg["role"], text, images))
    return "\n\n".join(system_bits), turns
