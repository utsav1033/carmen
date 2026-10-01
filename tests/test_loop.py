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


def test_json_parsing_and_structured_toggle(monkeypatch):
    from carmen import carmy
    assert carmy._parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    monkeypatch.setenv("CARMEN_STRUCTURED", "0")
    assert "format" not in carmy._request("s", "p", {}, "m", "high")["output_config"]


def test_dotenv_loads_without_overriding(tmp_path, monkeypatch):
    from carmen import dotenv
    f = tmp_path / ".env"
    f.write_text('# comment\nexport CARMEN_T1="quoted value"\nCARMEN_T2=plain # note\nCARMEN_T3=from_file\n')
    monkeypatch.delenv("CARMEN_T1", raising=False)
    monkeypatch.delenv("CARMEN_T2", raising=False)
    monkeypatch.setenv("CARMEN_T3", "from_shell")
    dotenv.load(f)
    import os
    assert os.environ["CARMEN_T1"] == "quoted value"
    assert os.environ["CARMEN_T2"] == "plain"
    assert os.environ["CARMEN_T3"] == "from_shell"
    monkeypatch.delenv("CARMEN_T1")
    monkeypatch.delenv("CARMEN_T2")


def test_auth_error_stops_the_run(tmp_path, monkeypatch):
    import pytest
    from carmen import carmy

    def bad_key(op, **kw):
        raise carmy.CarmyAuthError("rejected")

    with pytest.raises(carmy.CarmyAuthError):
        loop.run("softmax", rounds=3, k=2, adapter=FakeAdapter(), write_fn=bad_key, reflect_fn=no_reflect,
                 peak=PEAK, runs_dir=tmp_path / "runs", memory_dir=tmp_path / "mem")


def test_anthropic_base_url_proxy_is_detected(monkeypatch):
    from carmen import carmy
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.mycompany.dev")
    assert carmy.via_proxy()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    assert not carmy.via_proxy()


def test_lint_allows_size_regimes_but_not_exact_sizes():
    assert lint("For rows longer than 4096 elements, split the row across more threads per threadgroup.") is None
    assert "4097" in lint("When rows have 4097 elements, add a special loop for the ragged tail.")


def test_playbook_accepts_evidence_written_as_prose(tmp_path):
    pb = Playbook(tmp_path / "pb.json")
    pb.add([{"text": "Subtract the row max before exponentiating so large inputs cannot overflow.",
             "kind": "op", "evidence": ["attempt 0-1"]}], "softmax", verified_attempts={"0-1"})
    assert pb.lessons and next(iter(pb.lessons.values())).evidence == ["0-1"]


def test_carmy_sees_what_was_already_tried(tmp_path, monkeypatch):
    monkeypatch.setattr(ops.get("softmax"), "timing_shapes", ((64, 1000),))
    seen = []
    base = scripted_carmy(["good", "zeros", "zeros", "zeros"])

    def write(op, **kw):
        seen.append(kw["history"])
        return base(op, **kw)

    reflected = []
    loop.run("softmax", rounds=2, k=2, patience=5, adapter=FakeAdapter(), write_fn=write,
             reflect_fn=lambda op, s: reflected.append(s) or [], peak=PEAK,
             runs_dir=tmp_path / "runs", memory_dir=tmp_path / "mem")
    assert seen[0] == "" and seen[1] == ""
    assert "0-0" in seen[2] and "verified" in seen[2] and "64x1000 f32" in seen[2]
    assert "0-1" in seen[2] and "rejected" in seen[2]
    assert len(reflected) == 1  # round 0 found the first champion; round 1 proved nothing


def test_bench_compares_loop_and_best_of_n_on_the_same_budget(tmp_path, monkeypatch):
    from carmen import bench
    monkeypatch.setattr(ops.get("softmax"), "timing_shapes", ((64, 1000),))
    calls = []

    def run_fn(op, **kw):
        calls.append((kw["mode"], kw["rounds"], kw["k"], kw["patience"]))
        return loop.run(op, adapter=FakeAdapter(), write_fn=scripted_carmy(["good"] * 4), reflect_fn=no_reflect,
                        peak=PEAK, **kw)

    out = bench.run(["softmax"], repeats=1, rounds=2, k=2, root=tmp_path / "bench", run_fn=run_fn,
                    on_progress=lambda m: None)
    assert [c[0] for c in calls] == ["loop", "bon"] and all(c[3] > c[1] for c in calls)
    assert {r["mode"] for r in out["table"]} == {"loop", "bon"}
    assert all(len(r["curve"]) == 2 for r in out["results"])
    assert "best-of-N" in (tmp_path / "bench" / "bench.md").read_text()
