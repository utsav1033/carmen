from carmen import loop, ops
from carmen.backends import Kernel
from carmen.memory import Playbook, lint

from .fakes import FakeAdapter

PEAK = {"chip": "numpy", "peak_gbps": 100.0}


def scripted_carmy(sources):
    """Stands in for the model: returns a scripted kernel per call, in order."""
    calls = iter(sources)

    def write(op, **kw):
        k = Kernel(next(calls), configs=[{"TG": 256}], plan="scripted")
        return k, [], {"input_tokens": 0, "output_tokens": 0}, "prompt"
    return write


def no_reflect(op, summary):
    return []


def test_loop_repairs_then_keeps_champion(tmp_path, monkeypatch):
    monkeypatch.setattr(ops.get("softmax"), "timing_shapes", ((64, 1000),))
    write = scripted_carmy(["no_max_subtraction", "drops_last_element", "good", "good"])
    sm = loop.run("softmax", rounds=2, k=2, adapter=FakeAdapter(), write_fn=write, reflect_fn=no_reflect,
                  peak=PEAK, runs_dir=tmp_path / "runs", memory_dir=tmp_path / "mem")
    assert sm["first_round_verified"] == "0/2"
    assert sm["verified"] == 2 and sm["champion"] is not None
    assert sm["naive_pass_but_wrong"] >= 1
    assert sm["hidden_clean"] == "2/2"


def test_best_of_n_mode_runs_without_learning(tmp_path, monkeypatch):
    monkeypatch.setattr(ops.get("softmax"), "timing_shapes", ((64, 1000),))
    sm = loop.run("softmax", rounds=1, k=2, mode="bon", adapter=FakeAdapter(),
                  write_fn=scripted_carmy(["good", "zeros"]), reflect_fn=no_reflect, peak=PEAK,
                  runs_dir=tmp_path / "runs", memory_dir=tmp_path / "mem")
    assert sm["verified"] == 1
    assert not (tmp_path / "mem" / "playbook.json").exists()


def test_playbook_credits_only_judge_verdicts(tmp_path):
    pb = Playbook(tmp_path / "pb.json")
    pb.add([{"text": "Subtract the row max before exponentiating so large inputs cannot overflow.",
             "kind": "op", "evidence": ["0-1"]}], "softmax", verified_attempts={"0-1"})
    lid = next(iter(pb.lessons))
    pb.credit([lid], [lid], verified=True, op="softmax")
    pb.credit([lid], [lid], verified=True, op="softmax")
    assert pb.lessons[lid].u > 0 and pb.lessons[lid].status == "resident"


def test_playbook_rejects_unproven_and_shape_specific_lessons(tmp_path):
    pb = Playbook(tmp_path / "pb.json")
    rejected = pb.add([
        {"text": "Use vectorized float4 loads when rows are long enough.", "kind": "chip", "evidence": ["9-9"]},
        {"text": "When the row length is 4097 add a special tail loop at the end.", "kind": "op", "evidence": ["0-0"]},
    ], "softmax", verified_attempts={"0-0"})
    reasons = [r for _, r in rejected]
    assert any("no evidence" in r for r in reasons)
    assert any("specific size" in r for r in reasons)
    assert not pb.lessons


def test_lint_allows_hardware_numbers():
    assert lint("Use a threadgroup of 256 threads with simd_sum reductions for long rows.") is None
