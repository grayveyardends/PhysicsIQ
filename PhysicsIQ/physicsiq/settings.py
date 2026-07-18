"""settings.py — one place for every user-tunable knob.

We store settings in FreeCAD's own parameter database (the thing behind
Tools > Edit parameters) under BaseApp/Preferences/Mod/PhysicsIQ, so they
survive restarts and can be edited by hand without our UI.

Usage:
    from physicsiq.settings import S
    port = S.server_port()          # read (with default)
    S.set_auto_run(True)            # write
"""

import FreeCAD

_PARAM_PATH = "User parameter:BaseApp/Preferences/Mod/PhysicsIQ"


def _grp():
    # GetParameterGroupByPath creates the group if missing — safe to call always.
    return FreeCAD.ParamGet(_PARAM_PATH)


class _Settings:
    # llama-server
    def models_dir(self) -> str:
        from physicsiq import default_models_dir
        return _grp().GetString("ModelsDir", "") or default_models_dir()

    def server_port(self) -> int:
        return _grp().GetInt("ServerPort", 8735)

    def ctx_size(self) -> int:
        # Tokens of context llama-server keeps. 8192 fits master prompt +
        # RAG snippets + scene + a decent chat tail on the 4060's 8 GB.
        return _grp().GetInt("CtxSize", 8192)

    def gpu_layers(self) -> int:
        # 99 = "offload everything to GPU". Lower this if VRAM runs out
        # while a PINN is training on the same card.
        return _grp().GetInt("GpuLayers", 99)

    def last_model(self) -> str:
        return _grp().GetString("LastModel", "")

    def set_last_model(self, path: str):
        _grp().SetString("LastModel", path)

    # generation
    def temperature(self) -> float:
        # Low-ish: we want obedient code generation, not creative prose.
        return _grp().GetFloat("Temperature", 0.35)

    def max_tokens(self) -> int:
        # Reasoning models spend this SAME budget on hidden thinking before
        # writing a word of answer (DeepSeek-R1 burns 1.5-5k tokens musing).
        # Too small a cap = generation dies mid-thought and you get an empty
        # reply. NOTE the real ceiling is CtxSize minus the prompt (~2.5k
        # tokens), so at CtxSize 8192 you get ~5.7k no matter what's here.
        return _grp().GetInt("MaxTokens", 10000)

    # cloud cloud brains (the claude CLI, billed to the subscription)
    def cloud_models(self) -> list:
        """The cloud entries that appear in the model dropdown, alongside your
        ggufs. Edit CloudModels in the parameter editor to add/remove any
        alias the CLI accepts. NOTE: Opus is generally Max-only — picking it
        on a Pro plan fails with a clear message, it does not downgrade."""
        raw = _grp().GetString("CloudModels", "sonnet,opus,fable")
        return [m.strip() for m in raw.split(",") if m.strip()]

    def cloud_bin(self) -> str:
        return _grp().GetString("CloudBin", "claude")

    def cloud_timeout(self) -> int:
        return _grp().GetInt("CloudTimeout", 300)

    # visual feedback (LOCAL vision model — it runs too often to bill)
    def visual_check(self) -> bool:
        # After code runs, screenshot the viewport and ask the vision model
        # whether it matches the request. Needs a model with an mmproj.
        return _grp().GetBool("VisualCheck", True)

    def set_visual_check(self, value: bool):
        _grp().SetBool("VisualCheck", value)

    def max_visual_fixes(self) -> int:
        # How many times the vision model may say "FIX" before we stop and
        # let the human look. Without a cap, two models can argue forever.
        return _grp().GetInt("MaxVisualFixes", 2)

    # agent behaviour
    def auto_run(self) -> bool:
        # False = show Run button and wait for the human (the safe default
        # we agreed on). True = agent mode: execute generated code at once.
        return _grp().GetBool("AutoRun", False)

    def set_auto_run(self, value: bool):
        _grp().SetBool("AutoRun", value)

    def max_fix_attempts(self) -> int:
        # How many times a failed script's traceback is sent back to the
        # model for self-correction before we give up and tell the user.
        return _grp().GetInt("MaxFixAttempts", 3)

    # physics
    def pinn_python(self) -> str:
        # The venv that has JAX. NOT FreeCAD's python.
        import os
        return _grp().GetString(
            "PinnPython", os.path.expanduser("~/.venvs/usage/bin/python"))

    def yield_strength_mpa(self) -> float:
        # Default material: aluminium 6061-T6. Override per-run in the UI
        # or by telling the model the material in your prompt.
        return _grp().GetFloat("YieldMPa", 276.0)

    def safety_factor_target(self) -> float:
        return _grp().GetFloat("SFTarget", 2.0)


S = _Settings()
