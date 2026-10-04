"""--for <model>: the score is declared up front, judged at the model's shapes, a champion per regime."""
import json

import pytest

from carmen import carmy, cli, e2e, judge, loop, ops, target
from carmen.backends import Kernel
from carmen.judge import worker

from .fakes import FakeAdapter
from .test_loop import PEAK, no_reflect, scripted_carmy

TGT = target.make("mlx-community/Qwen2.5-0.5B-Instruct-4bit", 64, "bfloat16", prompt_tokens=32)


def test_judge_times_each_regime_in_the_models_dtype_against_mx_compile():
    req = judge.build_request("residual_rmsnorm", Kernel("good", configs=[{"TG": 32}, {"TG": 64}]),
                              round_seed=1, hidden_seed=3, peak=100.0, target=TGT.to_json())
    assert any(c["family"] == "target" and c["dtype"] == "bfloat16" for c in req["visible"])
    v = worker.evaluate(req, FakeAdapter())
    assert v["correct"] and set(v["regimes"]) == {"decode", "prefill"}
    d = v["regimes"]["decode"]
    assert d["shape"] == [1, 64] and d["dtype"] == "bfloat16" and "speedup_compiled" in d and d["config"]["TG"] in (32, 64)
    assert v["score"] == pytest.approx(v["speedup_compiled_geomean"])
    assert v["hidden"]["failed"] == 0
    fb = judge.feedback(v)
    assert "decode 1x64 bfloat16" in fb and "vs mx.compile" in fb and "score" in fb


def test_untargeted_runs_score_as_before():
    req = judge.build_request("softmax", Kernel("good"), round_seed=1, peak=100.0)
    req.update({"timing_shapes": [[64, 1000]], "timing_dtypes": ["float32"]})
    v = worker.evaluate(req, FakeAdapter())
    assert v["score"] == v["speedup_geomean"] and "regimes" not in v


def test_targeted_loop_keeps_a_champion_per_regime_and_tells_carmy_the_target(tmp_path):
    seen = []
    write = scripted_carmy(["good", "zeros", "good", "good"])

    def spy(op, **kw):
        seen.append(kw.get("target", ""))
        return write(op, **kw)
    sm = loop.run("residual_rmsnorm", rounds=2, k=2, patience=5, adapter=FakeAdapter(), write_fn=spy,
                  reflect_fn=no_reflect, peak=PEAK, runs_dir=tmp_path / "runs", memory_dir=tmp_path / "mem",
                  target=TGT.to_json())
    assert all("bfloat16" in t and "decode 1x64" in t for t in seen)
    assert set(sm["regime_champions"]) == {"decode", "prefill"} and sm["target"]["dtype"] == "bfloat16"
    assert sm["score"] > 0
    # the prompt carmy really gets says how it's scored
    p = carmy.prompt(ops.get("residual_rmsnorm"), chip="M4", peak=100, playbook="", champion=None, last=None,
                     variant=0, k=1, target=TGT.describe())
    assert "TARGET: This kernel is for" in p and "vs mx.compile" in p


def test_for_refuses_ops_without_a_model_shape():
    with pytest.raises(SystemExit, match="softmax isn't wired"):
        target.resolve("softmax", "qwen0.5b")


def test_holdout_flags_a_judge_the_model_disagrees_with():
    ok = e2e.Holdout("prefill", 512, 68.0, 52.0, 48.0, judge_vs_compiled=1.1)
    bad = e2e.Holdout("decode", 1, 13.9, 8.7, 17.3, judge_vs_compiled=1.15)
    assert ok.agrees and not bad.agrees and bad.vs_compiled < 1


def test_run_for_model_prints_regimes_and_the_in_model_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(target, "resolve", lambda op, model, prompt: TGT)
    monkeypatch.setattr(judge, "peak_gbps", lambda backend: PEAK)
    run_dir = tmp_path / "runs" / "20261004-000000"
    (run_dir / "attempts" / "0-1").mkdir(parents=True)
    (run_dir / "attempts" / "0-1" / "kernel.json").write_text(json.dumps(Kernel("k").to_json()))
    champs = {"decode": {"attempt": "0-1", "config": {"TG": 32}, "shape": [1, 64], "dtype": "bfloat16",
                         "speedup": 1.3, "speedup_compiled": 1.15},
              "prefill": {"attempt": "0-1", "config": {"TG": 64}, "shape": [32, 64], "dtype": "bfloat16",
                          "speedup": 1.2, "speedup_compiled": 1.1}}
    summary = {"attempts": 2, "compiled": 2, "verified": 2, "first_round_verified": "2/2", "naive_pass": 2,
               "naive_pass_but_wrong": 0, "hidden_clean": "2/2", "champion": "0-1", "speedup": 1.2,
               "target": TGT.to_json(), "regime_champions": champs, "run_dir": str(run_dir)}
    got = {}

    def fake_run(op, **kw):
        got.update(kw)
        return summary
    monkeypatch.setattr(loop, "run", fake_run)
    monkeypatch.setattr(e2e, "holdout", lambda name, ch, on_progress: [
        e2e.Holdout("decode", 1, 13.9, 8.7, 17.3, ch["decode"][3]),
        e2e.Holdout("prefill", 32, 68.0, 52.0, 48.0, ch["prefill"][3])])
    assert cli.main(["run", "residual_rmsnorm", "--for", "qwen0.5b", "--runs", str(tmp_path / "runs"),
                     "--memory", str(tmp_path / "mem")]) == 0
    out = capsys.readouterr().out
    assert got["target"]["dtype"] == "bfloat16"
    assert "champion per regime" in out and "judge was wrong" in out and "holds" in out
    saved = json.loads((run_dir / "summary.json").read_text())
    assert [h["agrees"] for h in saved["holdout"]] == [False, True]


def test_e2e_uses_each_regimes_champion(tmp_path):
    d = tmp_path / "20261004-000000"
    for a in ("0-1", "1-0"):
        (d / "attempts" / a).mkdir(parents=True)
        (d / "attempts" / a / "kernel.json").write_text(json.dumps(Kernel("src " + a).to_json()))
    (d / "events.jsonl").write_text(json.dumps({"type": "run_started", "op": "residual_rmsnorm"}) + "\n")
    (d / "summary.json").write_text(json.dumps({"champion": "1-0", "target": TGT.to_json(), "regime_champions": {
        "decode": {"attempt": "0-1", "config": {"TG": 32}}, "prefill": {"attempt": "1-0", "config": {"TG": 256}}}}))
    champs, run_id, _ = e2e.latest_regime_champions(tmp_path)
    assert champs["decode"][0].source == "src 0-1" and champs["prefill"][1] == {"TG": 256}
