"""
Only tests app.model.model's pure-logic helpers (no real model weights, no
GPU). `peft`/`transformers` are already CI dependencies (see ci.yml), which
pull in `torch` transitively — so importing this module is fine in CI even
though torch itself isn't listed there directly. Model *loading* itself
(`load_model`) is not covered here — see README "Testing" for why.
"""

from unittest.mock import MagicMock

from app.model.model import count_trainable_parameters


def test_count_trainable_parameters_uses_peft_accounting_when_available():
    """peft models expose get_nb_trainable_parameters(); it should be used
    directly rather than recomputed, since it correctly accounts for frozen/
    quantized base-model parameters."""
    fake_model = MagicMock()
    fake_model.get_nb_trainable_parameters.return_value = (1_000_000, 100_000_000)

    result = count_trainable_parameters(fake_model)

    assert result["trainable_params"] == 1_000_000
    assert result["total_params"] == 100_000_000
    assert result["trainable_pct"] == 1.0


def test_count_trainable_parameters_falls_back_to_manual_count():
    """A plain (non-peft) model has no get_nb_trainable_parameters, so the
    helper must fall back to iterating model.parameters()."""

    class FakeParam:
        def __init__(self, n, requires_grad):
            self._n = n
            self.requires_grad = requires_grad

        def numel(self):
            return self._n

    class FakeModel:
        def parameters(self):
            return [FakeParam(80, requires_grad=False), FakeParam(20, requires_grad=True)]

    result = count_trainable_parameters(FakeModel())

    assert result["trainable_params"] == 20
    assert result["total_params"] == 100
    assert result["trainable_pct"] == 20.0


def test_count_trainable_parameters_handles_zero_total_params():
    fake_model = MagicMock()
    fake_model.get_nb_trainable_parameters.return_value = (0, 0)

    result = count_trainable_parameters(fake_model)

    assert result["trainable_pct"] == 0.0
