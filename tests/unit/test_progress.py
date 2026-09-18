"""进度控制与可复现性工具测试。"""
from __future__ import annotations

import numpy as np

from zonal_reliability_fusion.utils import (
    NullProgress,
    environment_info,
    get_rng,
    is_main_process,
    make_progress,
    resolve_progress,
    set_seed,
)


def test_is_main_process_default_and_explicit_rank():
    assert is_main_process() is True
    assert is_main_process(0) is True
    assert is_main_process(1) is False


def test_progress_enabled_by_default():
    bar = make_progress(range(3), desc="unit")
    assert not isinstance(bar, NullProgress)
    assert list(bar) == [0, 1, 2]
    bar.close()


def test_no_progress_disables_output(capsys):
    bar = make_progress(range(3), desc="hidden", disable=True)
    assert isinstance(bar, NullProgress)
    assert list(bar) == [0, 1, 2]
    bar.update(1)
    bar.set_postfix({"loss": 0.1})
    bar.set_description("x")
    bar.close()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_non_main_process_is_silent():
    assert isinstance(make_progress(range(2), rank=1), NullProgress)
    assert isinstance(make_progress(range(2), rank=0), (NullProgress, object))


def test_null_progress_wraps_iterables():
    assert list(NullProgress(["a", "b"])) == ["a", "b"]
    assert list(NullProgress()) == []


def test_resolve_progress_combination():
    assert resolve_progress(False, True) is True
    assert resolve_progress(True, True) is False
    assert resolve_progress(False, False) is False


def test_set_seed_is_reproducible():
    set_seed(0)
    first = np.random.rand(4)
    set_seed(0)
    second = np.random.rand(4)
    assert np.allclose(first, second)


def test_get_rng_is_independent():
    rng_a = get_rng(7)
    rng_b = get_rng(7)
    assert np.allclose(rng_a.random(4), rng_b.random(4))


def test_environment_info_does_not_query_cuda():
    info = environment_info()
    assert "cuda_available" not in info  # 默认不查询 CUDA，避免在 GPU 被占用时产生副作用
    assert "torch" in info and "numpy" in info
