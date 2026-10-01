import numpy as np
import pytest

from carmen import judge, ops
from carmen.backends import Kernel
from carmen.judge import check

from .fakes import FakeAdapter

FAST = {"timing_shapes": [[64, 1000]], "timing_dtypes": ["float32"]}


def run(source, op="softmax", configs=None, hidden_seed=None):
    default = {"matmul": [{"TG": 256, "BM": 32, "BN": 32}], "attention": [{"TG": 64, "BQ": 1}]}.get(op, [{"TG": 256}])
    k = Kernel(source, configs=configs or default)
    req = judge.build_request(op, k, round_seed=1, hidden_seed=hidden_seed, peak=100.0)
    req.update(FAST)
    if ops.get(op).dims == 3:
        req["timing_shapes"] = [[64, 64, 96]]
    from carmen.judge import worker
    return worker.evaluate(req, FakeAdapter())


@pytest.mark.parametrize("op", ["softmax", "masked_softmax", "layernorm", "rmsnorm", "add_rmsnorm", "matmul", "attention"])
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


def test_bf16_is_in_every_battery_and_round_trips_exactly():
    from carmen.ops.base import Case, fuzz_cases, hidden_cases, to_bf16
    for spec in ops.OPS.values():
        assert any(c.dtype == "bfloat16" for c in spec.visible)
        assert any(c.dtype == "bfloat16" for c in hidden_cases(spec, 3))
        assert any(c.dtype == "bfloat16" for c in fuzz_cases(spec, 3))
        x = spec.materialize(Case("normal", 2, 300, "bfloat16", 1, "", 40 if spec.dims == 3 else 0))[spec.input_names[0]]
        assert np.array_equal(x, to_bf16(x))  # already exact bf16 values


def test_empty_cases_are_refused():
    from carmen.ops.base import Case
    with pytest.raises(ValueError):
        Case("normal", 0, 64, "float32", 1)


def test_matmul_square_only_bug_fools_the_naive_check_but_not_the_judge():
    v = run("wrong_leading_dim", "matmul")
    assert v["naive_pass"] and not v["correct"]


def test_matmul_configs_must_define_tiles():
    v = run("good", "matmul", configs=[{"TG": 256}])
    assert v["stage"] == "static" and "BM" in v["error"]


def test_matmul_launch_covers_every_tile():
    spec = ops.get("matmul")
    grid, tg = spec.grid(65, 63, {"TG": 128, "BM": 32, "BN": 16})
    assert grid == (4 * 128, 3, 1) and tg == (128, 1, 1)


def test_unroll_pragmas_are_allowed_other_pragmas_are_not():
    from carmen.backends import static_check
    assert static_check(Kernel("#pragma unroll\nfor (int i = 0; i < 4; ++i) {}")) is None
    assert "pragma" in static_check(Kernel("#pragma once\nint x;"))


def test_attention_cache_offset_bug_fools_the_naive_check_but_not_the_judge():
    v = run("cache_offset_ignored", "attention")
    assert v["naive_pass"] and not v["correct"]


def test_attention_draws_realistic_head_sizes_and_never_fewer_keys_than_queries():
    from carmen.ops.base import fuzz_cases, hidden_cases
    spec = ops.get("attention")
    for c in fuzz_cases(spec, 5) + hidden_cases(spec, 5):
        assert c.inner >= c.rows and 1 <= c.n <= 160


def test_reflector_lesson_limit_fits_real_ideas():
    from carmen.memory import lint
    assert lint("Stage A and B tiles through threadgroup memory with " + "x" * 450) is None
    assert "too long" in lint("y" * 800)
