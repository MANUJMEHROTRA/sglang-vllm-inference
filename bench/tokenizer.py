"""Token counting without dragging in torch.

Building a prompt of an exact token length needs exactly two operations:
text -> token ids, and ids -> text. It does not need a model, a framework, or
a GPU.

That distinction matters a great deal on Colab. ``from transformers import
AutoTokenizer`` imports the whole transformers package, which imports torch.
On a runtime where an engine install has just swapped the pinned torch build
under the running kernel, that import can fail outright with things like:

    ImportError: cannot import name '_chunk_or_narrow_cat' from 'torch._utils'

which is torch failing to import *its own* internals because the install is
half-replaced. The engine notebooks (01-04) drive a server over HTTP and have
no legitimate need for torch in the client process, so they should not be
taking that risk to count tokens.

The ``tokenizers`` package is the Rust tokenizer that transformers itself
wraps. It is already present wherever transformers is, it loads the same
``tokenizer.json`` from the Hub, and it imports nothing from torch. Using it
directly makes the whole class of failure impossible rather than merely
unlikely.

``PromptBuilder`` accepts either backend -- notebooks 00 and 06 legitimately
load models and already hold a transformers tokenizer, and passing one of
those still works.
"""

from __future__ import annotations

from typing import Any, List, Sequence


class _FastTokenizer:
    """Wraps ``tokenizers.Tokenizer`` -- Rust, no torch, no framework."""

    backend = "tokenizers"

    def __init__(self, tok: Any):
        self._t = tok

    def encode(self, text: str) -> List[int]:
        return self._t.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._t.decode(list(ids))


class _HFTokenizer:
    """Wraps a transformers ``PreTrainedTokenizer``."""

    backend = "transformers"

    def __init__(self, tok: Any):
        self._t = tok

    def encode(self, text: str) -> List[int]:
        return self._t.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self._t.decode(list(ids))


def load_tokenizer(model_id: str, prefer_fast: bool = True) -> Any:
    """Load a tokenizer for ``model_id``, avoiding torch when possible.

    Tries the ``tokenizers`` backend first (no torch import at all) and falls
    back to transformers only if that fails -- for instance for a repo that
    ships no ``tokenizer.json``, or a gated repo needing auth.

    The fallback prints why it happened. A silent fallback here would reintroduce
    the torch import the fast path exists to avoid, and you would only find out
    when an unrelated-looking ImportError surfaced three cells later.
    """
    if prefer_fast:
        try:
            from tokenizers import Tokenizer  # noqa: PLC0415

            return _FastTokenizer(Tokenizer.from_pretrained(model_id))
        except Exception as exc:  # noqa: BLE001 - the reason matters, print it
            print(f"[tokenizer] 'tokenizers' backend unavailable for {model_id}: "
                  f"{type(exc).__name__}: {exc}")
            print("[tokenizer] falling back to transformers, which imports torch")

    from transformers import AutoTokenizer  # noqa: PLC0415

    return _HFTokenizer(AutoTokenizer.from_pretrained(model_id))


def as_tokenizer(obj: Any) -> Any:
    """Normalise whatever we were handed to the two-method interface.

    Accepts an already-wrapped adapter, a ``tokenizers.Tokenizer``, or a
    transformers tokenizer, so callers never have to care which they have.
    """
    if isinstance(obj, (_FastTokenizer, _HFTokenizer)):
        return obj
    if type(obj).__module__.split(".")[0] == "tokenizers":
        return _FastTokenizer(obj)
    return _HFTokenizer(obj)
