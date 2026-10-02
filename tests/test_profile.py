import json

from carmen import cli, profile as prof

SPEC = prof.Spec("mlx-community/Qwen2.5-0.5B-Instruct-4bit", 896, 24, 14, 2, 64, 4864, 151936, 1e-6,
                 "bfloat16", 4, 64)


def steps():
    return [prof.Step("rmsnorm before attention", "rmsnorm", 10.0, 24),
            prof.Step("q, k, v projections", "matmul", 40.0, 24),
            prof.Step("residual add + rmsnorm", "add_rmsnorm", 12.0, 24),
            prof.Step("final norm + vocabulary projection", "matmul", 300.0, 1)]


def test_table_shares_add_up_and_mark_carmen_kernels():
    rows = prof.table(steps(), measured_ms=2.0)
    body, total = rows[:-1], rows[-1]
    assert abs(sum(r["share"] for r in body) - 1) < 1e-9
    assert body[0]["step"] == "q, k, v projections"  # 40 us x 24 layers is the biggest
    assert {r["kind"]: r["carmen"] for r in body}["add_rmsnorm"] == "add_rmsnorm"
    assert abs(total["ms_total"] - (10 * 24 + 40 * 24 + 12 * 24 + 300) / 1e3) < 1e-9


def test_by_kind_groups_matmuls():
    kinds = prof.by_kind(steps())
    assert list(kinds)[0] == "matmul" and abs(sum(kinds.values()) - 1) < 1e-9


def test_profile_command_prints_and_saves(tmp_path, monkeypatch, capsys):
    fake = prof.Profile(SPEC, 512, 128, 4000.0, 180.0, steps(), steps(), 4.2)
    monkeypatch.setattr(prof, "run", lambda *a, **k: fake)
    assert cli.main(["profile", "--runs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "180.0 tok/s" in out and "4-bit weights" in out and "add_rmsnorm" in out
    saved = json.loads(next(tmp_path.glob("profile-*.json")).read_text())
    assert saved["spec"]["hidden"] == 896 and len(saved["decode"]) == 4
