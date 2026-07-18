"""claude_cli.py — the cloud cloud backend: shells out to the `claude` CLI.

WHY A SUBPROCESS AND NOT THE HTTP API: the API needs an ANTHROPIC_API_KEY and
bills pay-per-token. The `claude` binary is already installed and logged in,
and it bills to the Claude **subscription** — which is the budget Gabi
actually has. So we drive it in headless mode and treat it as a very good
text-completion endpoint:

    claude -p                      print-and-exit (no interactive REPL)
      --model sonnet               (Opus is usually Max-only; this is a setting)
      --system-prompt "<ours>"     REPLACES Claude Code's own agent prompt —
                                   it must behave like our CAD model, not like
                                   a coding assistant
      --disallowed-tools …         no Bash/Edit/Read: it may ONLY answer with
                                   text + one ```python block, exactly the same
                                   contract the local gguf obeys. Same tools,
                                   same parser, better brain.
      --output-format stream-json  token-by-token, so the panel streams
      --strict-mcp-config          ignore whatever MCP servers the user has

BUDGET RULE (the reason this file is small and dumb): a cloud call costs a
real prompt from a limited plan, so NOTHING in the agent may call this on its
own. It happens only when the user ticks cloud for one message. Self-correction
retries, the visual check and the PINN loop are hard-wired to the local
backend — see agent/loop.py::_generate(cloud=False).

No Qt, no FreeCAD in here: runnable from a terminal, which is how it's tested.
"""

import json
import subprocess

from physicsiq.llm.backends.base import (GenerationCancelled,
                                         decode_image_part, flatten_messages)

# Claude Code's own tools. We are using the model as a brain, not as an agent:
# it must not read the disk or run shell commands behind our back. Its ONLY
# way to affect anything is the python block we choose to execute.
_NO_TOOLS = ("Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task,"
             "NotebookEdit,TodoWrite,SlashCommand")


class CloudUnavailable(RuntimeError):
    """The CLI is missing, not logged in, or the plan refuses the model."""


class ClaudeCliClient:
    """Same chat_stream() signature as LlamaClient — ui/workers.py and
    agent/loop.py cannot tell the two apart."""

    def __init__(self, model="sonnet", binary="claude", cwd=None,
                 timeout=300):
        self.model = model
        self.binary = binary
        self.cwd = cwd
        self.timeout = timeout
        # Filled in after each call so the panel can show what a tick cost.
        self.last_cost_usd = None
        self.last_error = None

    def _argv(self, system: str, stream_json_input: bool):
        argv = [
            self.binary, "-p",
            "--model", self.model,
            "--output-format", "stream-json",
            "--include-partial-messages",   # token deltas, not just the end
            "--verbose",                    # required for stream-json under -p
            "--strict-mcp-config",          # ignore the user's MCP servers
            "--exclude-dynamic-system-prompt-sections",  # no CLAUDE.md, no env
            "--disallowed-tools", _NO_TOOLS,
        ]
        if system:
            # REPLACES Claude Code's agent prompt (not --append-system-prompt):
            # we want our CAD persona, not a coding assistant's.
            argv += ["--system-prompt", system]
        if stream_json_input:
            argv += ["--input-format", "stream-json"]
        return argv

    def _stdin_payload(self, turns, stream_json_input: bool) -> str:
        """What we feed the CLI on stdin.

        Text mode (the common case — planning has no images): the whole
        conversation as a plain transcript. `claude -p` reads its prompt from
        stdin, which also dodges argv length limits on a long history.

        stream-json mode (only when images are attached): one JSON user
        message carrying real content blocks, because a PNG cannot be a
        transcript line.
        """
        images = [img for _role, _text, imgs in turns for img in imgs]
        transcript = self._transcript(turns)
        if not stream_json_input:
            return transcript + "\n"

        content = [{"type": "text", "text": transcript}]
        for part in images:
            mime, data = decode_image_part(part)
            content.append({"type": "image",
                            "source": {"type": "base64",
                                       "media_type": mime, "data": data}})
        msg = {"type": "user",
               "message": {"role": "user", "content": content}}
        return json.dumps(msg) + "\n"

    @staticmethod
    def _transcript(turns) -> str:
        """History -> a plain transcript. The CLI takes ONE prompt, not a
        message array, so past turns become labelled text. The last user turn
        is left unlabelled so it reads as the actual question."""
        if not turns:
            return ""
        lines = []
        for role, text, imgs in turns[:-1]:
            who = "User" if role == "user" else "Assistant"
            lines.append(f"{who}: {text}")
        role, text, imgs = turns[-1]
        if lines:
            lines.append("")
        lines.append(text)
        if imgs:
            lines.append(f"\n({len(imgs)} image(s) attached above.)")
        return "\n\n".join(lines)

    def chat_stream(self, messages, temperature=None, max_tokens=None,
                    grammar=None, on_chunk=None, on_thinking=None,
                    should_cancel=None) -> str:
        """temperature/max_tokens/grammar are accepted and IGNORED — the CLI
        exposes none of them. Keeping the signature identical is what lets the
        rest of the app stay backend-blind."""
        self.last_cost_usd = None
        self.last_error = None

        system, turns = flatten_messages(messages)
        has_images = any(imgs for _r, _t, imgs in turns)
        argv = self._argv(system, stream_json_input=has_images)

        try:
            proc = subprocess.Popen(
                argv, cwd=self.cwd, text=True, bufsize=1,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise CloudUnavailable(
                f"'{self.binary}' not found — is Claude Code installed?") from exc

        try:
            proc.stdin.write(self._stdin_payload(turns, has_images))
            proc.stdin.close()          # EOF = "that's the whole prompt"
            answer = self._read_stream(proc, on_chunk, on_thinking,
                                       should_cancel)
        except GenerationCancelled:
            proc.kill()
            proc.wait()
            raise
        finally:
            if proc.stdout:
                proc.stdout.close()

        stderr = (proc.stderr.read() or "").strip() if proc.stderr else ""
        code = proc.wait()
        if self.last_error:
            raise CloudUnavailable(self.last_error)
        if code != 0 and not answer:
            # The usual causes: not logged in, rate limit hit, or the plan
            # does not include this model (Opus on Pro).
            raise CloudUnavailable(
                f"claude exited {code} (model '{self.model}'): "
                f"{stderr[-300:] or 'no output'}")
        return answer

    def _read_stream(self, proc, on_chunk, on_thinking, should_cancel):
        """Parse the CLI's JSON-lines output.

        Shapes we care about (everything else is skipped, by design — the CLI
        adds new event types over time and none of them should break us):

          {"type":"stream_event","event":{"type":"content_block_delta",
                                          "delta":{"type":"text_delta",
                                                   "text":"Hel"}}}
          {"type":"assistant","message":{"content":[{"type":"text",...}]}}
          {"type":"result","subtype":"success","result":"<final text>",
           "total_cost_usd":0.01}
        """
        streamed, final = [], None
        for line in proc.stdout:
            if should_cancel is not None and should_cancel():
                raise GenerationCancelled()
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = event.get("type")
            if kind == "stream_event":
                delta = (event.get("event") or {}).get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    piece = delta.get("text") or ""
                    if piece:
                        streamed.append(piece)
                        if on_chunk is not None:
                            on_chunk(piece)
                elif dtype == "thinking_delta":
                    thought = delta.get("thinking") or ""
                    if thought and on_thinking is not None:
                        on_thinking(thought)

            elif kind == "result":
                self.last_cost_usd = event.get("total_cost_usd")
                if event.get("subtype") == "success":
                    final = event.get("result") or None
                else:
                    self.last_error = (
                        event.get("result")
                        or f"claude returned '{event.get('subtype')}'")

        # The `result` line is authoritative (it is the complete message). Fall
        # back to what we streamed if the CLI ever stops sending one.
        return (final if final is not None else "".join(streamed)).strip()

    def chat(self, messages, **kw) -> str:
        return self.chat_stream(messages, **kw)
