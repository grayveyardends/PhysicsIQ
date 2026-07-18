"""LLM backends. One file per way of reaching a model.

    local.py       llama-server on localhost (the gguf in models/) — the
                   hackathon demo runs entirely on this.
    claude_cli.py  the `claude` CLI, billed to the Claude subscription — the
                   cloud button, for the thinking the small model can't do.

They share ONE message shape and ONE method (see base.py), so nothing above
this folder knows which model answered. A future custom engine (Inferio) is a
third file here and nothing else changes.
"""
