"""client.py — the ONE place the rest of the app asks for a model.

Everything else imports from here and never from llm/backends/* directly, so
"which model answers?" is decided in exactly one function:

    get_client(cloud=False) -> LlamaClient        (local gguf, llama-server)
    get_client(cloud=True)  -> ClaudeCliClient    (cloud the claude CLI)

Both expose the same chat_stream(...) — see llm/backends/base.py for the
interface and the message shape.

BUDGET RULE, worth repeating because it is a money question: `cloud=True` is
only ever passed when the user ticked cloud for that one message. Every loop in
the agent (self-correction retries, the visual check, the PINN fix loop) calls
_generate() with the default cloud=False, so no amount of agent enthusiasm can
spend the subscription on its own.
"""

from physicsiq.llm.backends.base import (GenerationCancelled,  # noqa: F401
                                         image_part, load_image_file,
                                         split_thinking)
from physicsiq.llm.backends.claude_cli import (ClaudeCliClient,  # noqa: F401
                                               CloudUnavailable)
from physicsiq.llm.backends.local import LlamaClient  # noqa: F401


def get_client(cloud: bool = False, model: str = None):
    """model: the cloud alias ('sonnet', 'opus', …) chosen in the dropdown.
    Ignored for the local backend, which serves whatever llama-server loaded."""
    if cloud:
        from physicsiq import REPO_DIR
        from physicsiq.settings import S
        return ClaudeCliClient(model=model or "sonnet", binary=S.cloud_bin(),
                               cwd=REPO_DIR, timeout=S.cloud_timeout())
    from physicsiq.llm.server import MANAGER
    return LlamaClient(MANAGER.base_url())
