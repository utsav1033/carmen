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
    fake = [T("stock: x + r, then rms_norm", 1, 10.0, 4.0), T("carmen, as e2e calls it", 1, 40.0, 30.0),
            T("carmen, prebuilt + mx.compile", 1, float("nan"), float("nan"), "ValueError: nope"),
            T("stock: x + r, then rms_norm", 512, 20.0, 4.0), T("carmen, as e2e calls it", 512, 25.0, 30.0)]
    monkeypatch.setattr(e2e, "call_bench", lambda *a, **k: fake)
    assert cli.main(["e2e", "--calls", "--runs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "0.25×" in out and "ValueError: nope" in out and "+1.44 ms per word" in out
    assert next(tmp_path.glob("calls-*.json"))
