import numpy as np
import pytest

from carmen import judge, ops
from carmen.backends import Kernel
from carmen.judge import check

from .fakes import FakeAdapter

FAST = {"timing_shapes": [[64, 1000]], "timing_dtypes": ["float32"]}


def run(source, op="softmax", configs=None, hidden_seed=None):
    k = Kernel(source, configs=configs or [{"TG": 256}])
    req = judge.build_request(op, k, round_seed=1, hidden_seed=hidden_seed, peak=100.0)
    req.update(FAST)
    from carmen.judge import worker
    return worker.evaluate(req, FakeAdapter())


@pytest.mark.parametrize("op", ["softmax", "masked_softmax", "layernorm", "rmsnorm", "add_rmsnorm"])
def test_correct_kernel_passes_everything(op):
    v = run("good", op, hidden_seed=7)
    assert v["correct"], v["configs"][0]["failures"]
    assert v["naive_pass"]
    assert v["hidden"]["failed"] == 0
    assert v["timing"] and v["speedup_geomean"] > 0


def test_missing_max_subtraction_passes_naive_check_but_not_the_judge():
    v = run("no_max_subtraction")
    assert v["naive_pass"]            # a KernelBench-style check is fooled
    assert not v["correct"]
    f = v["configs"][0]["failures"][0]
    assert "NaN" in f["patterns"][0]


def test_unwritten_tail_is_located_and_shrunk():
    v = run("drops_last_element")
    f = v["configs"][0]["failures"][0]
    assert any("NaN" in p for p in f["patterns"])
    assert f["smallest_failing"] is not None


def test_fp16_store_escapes_naive_but_not_the_judge():
    v = run("stores_through_fp16")
    assert v["naive_pass"] and not v["correct"]


def test_all_zero_softmax_is_caught():
    v = run("zeros")
    assert not v["correct"]


def test_race_is_caught_by_repeat_runs():
    v = run("racy")
    assert not v["correct"]


def test_write_past_end_is_caught():
    f = run("writes_past_end")["configs"][0]["failures"][0]
    assert "wrote past the end of the output" in f["patterns"]


def test_autotune_drops_incorrect_configs():
    v = run("bad_when_big_tg", configs=[{"TG": 1024}, {"TG": 256}])
    assert v["correct"] and v["best_config"] == {"TG": 256}
    assert not v["configs"][0]["passed"]


def test_compile_error_is_reported():
    v = run("does_not_exist")
    assert v["stage"] == "compile" and "undeclared" in v["error"]


def test_static_check_rejects_bad_configs():
    assert run("good", configs=[{"TG": 100}])["stage"] == "static"


def test_feedback_never_mentions_hidden_tests():
    v = run("good", hidden_seed=3)
    text = judge.feedback(v)
    assert "hidden" not in text.lower() and "Speed" in text


def test_tolerance_is_relative_and_capped():
    assert check.tolerance("float32", 4096, 0.0) == check.floor_tol("float32", 4096)
    assert check.tolerance("float32", 4096, 1.0) == check.CAP * check.floor_tol("float32", 4096)


def test_locate_tail():
    bad = np.zeros((4, 100), bool)
    bad[:, -1] = True
    assert "tail" in check.locate(bad)


def test_reference_rows_sum_to_one():
    spec = ops.get("softmax")
    for case in spec.visible[:6]:
        ref = spec.reference(spec.materialize(case))
        finite = np.isfinite(ref).all(-1)
        assert np.allclose(ref[finite].sum(-1), 1.0)


def test_strict_naive_check_is_reported_and_harder_to_fool():
    v = run("stores_through_fp16")
    assert v["naive_pass"] and "naive_pass_strict" in v


def test_locate_reports_counts_for_tiny_fractions():
    bad = np.zeros((3, 5000), bool)
    bad[1, 17] = True
    assert "1 element," in check.locate(bad)


@pytest.mark.parametrize("op", ["layernorm", "rmsnorm", "add_rmsnorm"])
def test_norms_reject_an_empty_kernel(op):
    assert not run("zeros", op)["correct"]


def test_layernorm_catches_one_pass_variance():
    v = run("one_pass_variance", "layernorm")
    assert not v["correct"]
    assert any(f["case"].startswith("offset") for f in v["configs"][0]["failures"])


def test_every_op_has_a_golden_kernel_and_mutants_that_apply():
    from carmen import broken
    for name in ops.OPS:
        ms = broken.mutants(name)
        assert ms and all(k.source != broken.golden(name).source for _, k in ms)
