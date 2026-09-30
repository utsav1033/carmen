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


def test_litellm_client_and_json_parsing(monkeypatch):
    from carmen import carmy
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://proxy:4000")
    assert carmy.via_litellm() and str(carmy._client().base_url).startswith("http://proxy:4000")
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


def test_route_masks_the_key(monkeypatch):
    from carmen import carmy
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secretsecret1234")
    r = carmy.route()
    assert "…1234" in r and "secret" not in r


def test_anthropic_base_url_proxy_is_detected(monkeypatch):
    from carmen import carmy
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.mycompany.dev")
    assert carmy.via_proxy() and "proxy at https://llm.mycompany.dev" in carmy.route()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    assert not carmy.via_proxy()
