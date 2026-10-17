"""
Tests for app.utils.environment. `torch` is deliberately NOT a CI dependency
(see .github/workflows/ci.yml) since it's heavy and unnecessary for the rest
of the unit test suite — so these tests inject a fake `torch` module into
sys.modules rather than requiring the real package. `environment.py` imports
torch lazily inside each function body specifically so this works.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_torch(monkeypatch):
    """Installs a minimal fake `torch` module for the duration of a test."""
    fake = types.ModuleType("torch")
    fake.cuda = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def _reimport_environment():
    """environment.py has no module-level torch import, so a plain import is
    enough — but ensure we're not using a stale cached module either way."""
    sys.modules.pop("app.utils.environment", None)
    from app.utils import environment

    return environment


def test_require_cuda_raises_actionable_error_when_no_gpu(fake_torch):
    fake_torch.cuda.is_available.return_value = False
    environment = _reimport_environment()

    with pytest.raises(RuntimeError, match="No CUDA GPU detected"):
        environment.require_cuda()


def test_require_cuda_error_mentions_colab_docs(fake_torch):
    """The error should point at the actual fix for the common case (no local
    NVIDIA GPU), not just state the problem."""
    fake_torch.cuda.is_available.return_value = False
    environment = _reimport_environment()

    with pytest.raises(RuntimeError, match="COLAB.md"):
        environment.require_cuda()


def test_require_cuda_passes_silently_when_gpu_available(fake_torch):
    fake_torch.cuda.is_available.return_value = True
    environment = _reimport_environment()

    environment.require_cuda()  # should not raise


def test_describe_gpu_reports_no_gpu_message(fake_torch):
    fake_torch.cuda.is_available.return_value = False
    environment = _reimport_environment()

    assert environment.describe_gpu() == "No CUDA GPU available."


def test_describe_gpu_reports_name_and_vram(fake_torch):
    fake_props = MagicMock()
    fake_props.total_memory = 16 * (1024**3)  # 16 GB
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.current_device.return_value = 0
    fake_torch.cuda.get_device_name.return_value = "Tesla T4"
    fake_torch.cuda.get_device_properties.return_value = fake_props

    environment = _reimport_environment()
    result = environment.describe_gpu()

    assert "Tesla T4" in result
    assert "16.0 GB" in result


def test_gpu_memory_stats_returns_zeros_without_gpu(fake_torch):
    fake_torch.cuda.is_available.return_value = False
    environment = _reimport_environment()

    stats = environment.gpu_memory_stats()

    assert stats == {
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "max_allocated_gb": 0.0,
        "max_reserved_gb": 0.0,
    }


def test_gpu_memory_stats_converts_bytes_to_gb(fake_torch):
    gb = 1024**3
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.memory_allocated.return_value = 2 * gb
    fake_torch.cuda.memory_reserved.return_value = 3 * gb
    fake_torch.cuda.max_memory_allocated.return_value = 4 * gb
    fake_torch.cuda.max_memory_reserved.return_value = 5 * gb

    environment = _reimport_environment()
    stats = environment.gpu_memory_stats()

    assert stats["allocated_gb"] == 2.0
    assert stats["reserved_gb"] == 3.0
    assert stats["max_allocated_gb"] == 4.0
    assert stats["max_reserved_gb"] == 5.0


def test_reset_peak_memory_stats_calls_torch_when_gpu_available(fake_torch):
    fake_torch.cuda.is_available.return_value = True
    environment = _reimport_environment()

    environment.reset_peak_memory_stats()

    fake_torch.cuda.reset_peak_memory_stats.assert_called_once()


def test_reset_peak_memory_stats_is_noop_without_gpu(fake_torch):
    fake_torch.cuda.is_available.return_value = False
    environment = _reimport_environment()

    environment.reset_peak_memory_stats()  # should not raise

    fake_torch.cuda.reset_peak_memory_stats.assert_not_called()


def test_supports_bf16_false_without_gpu(fake_torch):
    fake_torch.cuda.is_available.return_value = False
    environment = _reimport_environment()

    assert environment.supports_bf16() is False


def test_supports_bf16_true_on_ampere_plus(fake_torch):
    """Regression test for the real bug hit on Colab's free-tier T4: bf16=True
    was hardcoded and raised a hard error on pre-Ampere GPUs."""
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.is_bf16_supported.return_value = True
    environment = _reimport_environment()

    assert environment.supports_bf16() is True


def test_supports_bf16_false_on_pre_ampere_gpu(fake_torch):
    """E.g. a Tesla T4 (Turing, compute capability 7.5) — CUDA is available
    but bf16 specifically is not supported in hardware."""
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.is_bf16_supported.return_value = False
    environment = _reimport_environment()

    assert environment.supports_bf16() is False
