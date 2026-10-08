import json

from carmen import cli, e2e
from carmen.backends import Kernel


def _run_dir(root, name, op, champion):
    d = root / name
    (d / "attempts" / "0-1").mkdir(parents=True)
    (d / "events.jsonl").write_text(json.dumps({"type": "run_started", "op": op}) + "\n")
    (d / "summary.json").write_text(json.dumps({"champion": champion}))
    (d / "attempts" / "0-1" / "kernel.json").write_text(json.dumps(Kernel("src " + name).to_json()))
    (d / "attempts" / "0-1" / "verdict.json").write_text(json.dumps({"best_config": {"TG": 128}, "timing": []}))


def test_latest_champion_picks_the_newest_run_of_the_right_op(tmp_path):
    _run_dir(tmp_path, "20261001-000000", "residual_rmsnorm", "0-1")
    _run_dir(tmp_path, "20261002-000000", "residual_rmsnorm", "0-1")
    _run_dir(tmp_path, "20261003-000000", "softmax", "0-1")
    _run_dir(tmp_path, "20261004-000000", "residual_rmsnorm", None)
    kernel, config, kid, _ = e2e.latest_champion(tmp_path)
    assert kernel.source == "src 20261002-000000" and config == {"TG": 128} and kid.startswith("20261002")


def test_e2e_command_reports_speed_and_answers(tmp_path, monkeypatch, capsys):
    _run_dir(tmp_path, "20261002-000000", "residual_rmsnorm", "0-1")
    fake = [e2e.Result("stock", 4000, 150.0, tokens_total=128),
            e2e.Result("kernel", 4100, 165.0, 128, 128, 0.02, True),
            e2e.Result("both", 4200, 172.0, 90, 128, 0.03, True)]
    monkeypatch.setattr(e2e, "run", lambda *a, **k: fake)
    assert cli.main(["e2e", "--runs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1.10×" in out and "same 128 tokens" in out and "diverge at token 91" in out
    assert next(tmp_path.glob("e2e-*.json"))
    assert e2e.verdict_lines(fake)[0] == "kernel: decode +10.0% vs stock, same answers"


def test_calls_command_prints_where_the_time_goes(tmp_path, monkeypatch, capsys):
    _run_dir(tmp_path, "20261002-000000", "residual_rmsnorm", "0-1")
    T = e2e.CallTiming
    fake = [T("stock: x + r, then rms_norm", 1, 10.0, 4.0), T("carmen, compiled call (what e2e uses)", 1, 40.0, 30.0),
            T("carmen, prebuilt + mx.compile", 1, float("nan"), float("nan"), "ValueError: nope"),
            T("stock: x + r, then rms_norm", 512, 20.0, 4.0), T("carmen, compiled call (what e2e uses)", 512, 25.0, 30.0)]
    monkeypatch.setattr(e2e, "call_bench", lambda *a, **k: fake)
    assert cli.main(["e2e", "--calls", "--runs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "0.25×" in out and "ValueError: nope" in out and "+1.44 ms per word" in out
    assert next(tmp_path.glob("calls-*.json"))


def test_modes_combine_and_typos_fail_early():
    assert e2e.parts("stock") == set() and e2e.parts("all") == {"plumbing", "compile", "kernel"}
    assert e2e.parts("plumbing+compile") == {"plumbing", "compile"} and e2e.parts("both") == {"plumbing", "kernel"}
    import pytest
    with pytest.raises(SystemExit, match="compiel"):
        e2e.parts("plumbing+compiel")


def test_beyond_noise_needs_ranges_that_dont_overlap():
    R = e2e.Result
    stock = R("stock", 2000, 150, decode_lo=145, decode_hi=155)
    assert e2e.beyond_noise(R("a", 2000, 165, decode_lo=158, decode_hi=170), stock) == "faster"
    assert e2e.beyond_noise(R("b", 2000, 160, decode_lo=150, decode_hi=168), stock) == "within noise"
    assert e2e.beyond_noise(R("c", 2000, 130, decode_lo=125, decode_hi=140), stock) == "slower"
    assert round(e2e.speed_limit_tps(300_000_000, 120)) == 400


def test_e2e_prints_noise_and_the_speed_limit(tmp_path, monkeypatch, capsys):
    from carmen import judge
    R = e2e.Result
    fake = [R("stock", 2000, 150.0, tokens_total=128, decode_lo=145, decode_hi=155, turns=5, weight_bytes=300_000_000),
            R("compile", 2100, 170.0, 128, 128, 0.0, True, decode_lo=160, decode_hi=175, turns=5)]
    monkeypatch.setattr(e2e, "run", lambda *a, **k: fake)
    monkeypatch.setattr(judge, "peak_gbps", lambda backend: {"chip": "M4", "peak_gbps": 120.0})
    assert cli.main(["e2e", "--modes", "stock,compile", "--runs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "faster" in out and "~400 tok/s" in out and "stock reaches 38%" in out and "compile 42%" in out
    saved = json.loads(next(tmp_path.glob("e2e-*.json")).read_text())
    assert saved["decode_speed_limit_tps"] == 400.0


def test_holdout_says_tie_or_unsteady_instead_of_guessing():
    H = e2e.Holdout
    assert H("decode", 1, 13.9, 17.1, 17.6, 1.02).verdict == "tie"  # 0.97x: within 5%
    assert H("decode", 1, 13.9, 17.1, 18.1, 1.02).verdict == "disagrees"  # 0.94x with no range to say otherwise
    assert H("decode", 1, 13.9, 17.1, 18.1, 1.02, ratio_lo=0.9, ratio_hi=1.03).verdict == "tie"
    assert H("prefill", 512, 25.7, 33.6, 59.7, 1.52, 0.5, 0.6, stock_spread=1.4).verdict == "unsteady"
    assert H("prefill", 512, 25.7, 33.6, 59.7, 1.52, 0.5, 0.6).verdict == "disagrees"


def test_paired_turns_see_a_win_that_overlapping_ranges_hide():
    R = e2e.Result
    stock_turns = [150, 130, 160, 140, 155, 135, 145, 150]  # the Mac swings +-10% between turns
    fast_turns = [t * 1.05 for t in stock_turns]  # every turn 5% faster than the stock run beside it
    stock = R("stock", 2000, 147.5, decode_lo=130, decode_hi=160, decode_turns=stock_turns)
    mlp = R("mlp", 2000, 154.9, decode_lo=136.5, decode_hi=168, decode_turns=fast_turns)
    lo, hi = e2e.paired(mlp, stock)[1:]
    assert e2e.beyond_noise(mlp, stock) == "faster" and 1.0 < lo <= hi
    flat = R("x", 2000, 147.5, decode_lo=130, decode_hi=160, decode_turns=[150, 140, 160, 125, 160, 140, 140, 152])
    assert e2e.beyond_noise(flat, stock) == "within noise"


def test_bundled_champions_load_and_pass_static_checks():
    from carmen import champions, ops
    from carmen.backends import static_check
    assert champions.available() == ["qwen2.5-0.5b-4bit"]
    ks = champions.kernels("qwen2.5-0.5b-4bit")
    assert set(ks) == {"mlp_up", "mlp_down"}
    for op, (k, cfg, origin) in ks.items():
        assert static_check(k) is None and ops.get(op).check_config(cfg) is None and cfg in k.configs and origin
    assert champions.for_model("mlx-community/Qwen2.5-0.5B-Instruct-4bit") == "qwen2.5-0.5b-4bit"


def test_speedup_runs_stock_compile_and_the_bundled_kernels(tmp_path, monkeypatch, capsys):
    from carmen import judge
    got = {}
    R = e2e.Result

    def fake_run(name, modes, *a, mlp_kernels=None, **k):
        got.update(name=name, modes=modes, mlp=mlp_kernels, repeats=a[4] if len(a) > 4 else None)
        return [R("stock", 2000, 150.0, tokens_total=256, decode_lo=149, decode_hi=151, turns=30,
                  decode_turns=[150.0] * 30),
                R("mlp+compile", 2000, 155.0, 256, 256, 0.0, True, decode_lo=154, decode_hi=156, turns=30,
                  decode_turns=[155.0] * 30)]
    monkeypatch.setattr(e2e, "run", fake_run)
    monkeypatch.setattr(judge, "peak_gbps", lambda backend: {"chip": "M4", "peak_gbps": 96.0})
    assert cli.main(["speedup", "--runs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert got["name"] == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    assert got["modes"] == ["stock", "compile", "mlp+compile"] and set(got["mlp"]) == {"mlp_up", "mlp_down"}
    assert "published: 1.03x" in out and "faster" in out


def test_offline_retry_only_on_network_errors(monkeypatch):
    import os
    from carmen.profile import offline_retry
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls = []

    def flaky(name):
        calls.append(os.environ.get("HF_HUB_OFFLINE"))
        if len(calls) == 1:
            raise RuntimeError("Server disconnected without sending a response.")
        return "model"
    assert offline_retry(flaky)("m") == "model" and calls == [None, "1"]
    import pytest

    def broken(name):
        raise ValueError("bad config")
    with pytest.raises(ValueError):
        offline_retry(broken)("m")
