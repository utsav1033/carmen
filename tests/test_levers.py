import pytest

from carmen import cli, judge, levers


def test_draft_and_8bit_source_come_from_the_model_name():
    assert levers.default_draft("mlx-community/Qwen2.5-3B-Instruct-4bit") == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    assert levers.default_draft("mlx-community/Qwen2.5-0.5B-Instruct-4bit") is None  # no smaller sibling
    assert levers.default_draft("mlx-community/Mistral-7B-Instruct-4bit") is None
    assert levers.eight_bit("mlx-community/Qwen2.5-3B-Instruct-4bit") == "mlx-community/Qwen2.5-3B-Instruct-8bit"
    assert levers.eight_bit("mlx-community/Qwen2.5-3B-Instruct-bf16") is None


class _Chars:
    def encode(self, text):
        return [ord(c) for c in text]


def test_prompt_is_long_varied_text_not_a_repeated_sentence():
    ids = levers.prompt_ids(_Chars(), 32768)
    assert len(ids) == 32768 and len(set(ids)) > 50
    assert ids[:200] != ids[200:400]
    with pytest.raises(SystemExit):
        levers.prompt_ids(_Chars(), 10**8)


def test_speed_limit_counts_the_kv_cache():
    assert levers.speed_limit_tps(int(1e9), 0, 100.0) == pytest.approx(100.0)
    assert levers.speed_limit_tps(int(1e9), int(1e9), 100.0) == pytest.approx(50.0)


def test_drift_is_zero_for_identical_odds_and_grows_when_they_move():
    mx = pytest.importorskip("mlx.core")
    ref = mx.log(mx.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]))
    assert levers.drift(mx, ref, ref) == pytest.approx((1.0, 0.0), abs=1e-6)
    top1, kl = levers.drift(mx, ref, mx.log(mx.array([[0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])))
    assert top1 == pytest.approx(0.5) and kl > 0.1


def test_requantize_rewrites_every_quantized_layer_at_3_bits():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    model = nn.Sequential(nn.QuantizedLinear(128, 64, bias=False), nn.QuantizedLinear(64, 128, bias=False))
    x = mx.random.normal((1, 128))
    before = model(x)
    assert levers.requantize(mx, nn, model, bits=3) == 2
    assert all(m.bits == 3 for m in model.layers)
    after = model(x)
    assert after.shape == before.shape and 0 < float(mx.abs(after - before).max().item()) < float(mx.abs(before).max().item())


def _r(mode, ctx, dec, **kw):
    return levers.LeverResult(mode, 500.0, dec[0], tokens_total=128, decode_turns=dec, prefill_turns=[500.0] * len(dec),
                              context=ctx, weight_bytes=int(1.74e9), kv_bytes=int(0.1e9 if ctx > 1000 else 0.01e9),
                              turns=len(dec), decode_lo=min(dec), decode_hi=max(dec), **kw)


def test_levers_command_prints_each_context_and_the_comparison(tmp_path, monkeypatch, capsys):
    fake = [_r("stock", 512, [49.0, 50.0, 48.0]),
            _r("spec", 512, [80.0, 82.0, 79.0], tokens_match=128, accept_rate=0.62),
            _r("kv4", 512, [49.0, 50.5, 48.0], tokens_match=40, top1=0.97, kl=0.021),
            _r("stock", 8192, [40.0, 41.0, 39.0]),
            _r("spec", 8192, [60.0, 61.0, 59.0], tokens_match=128, accept_rate=0.60),
            _r("kv4", 8192, [46.0, 47.0, 45.0], tokens_match=30, top1=0.95, kl=0.040)]
    monkeypatch.setattr(levers, "run", lambda *a, **k: fake)
    monkeypatch.setattr(judge, "peak_gbps", lambda backend: {"chip": "M4", "peak_gbps": 94.0})
    assert cli.main(["levers", "qwen3b", "--levers", "stock,spec,kv4", "--runs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "8,192 prompt tokens" in out and "decode vs stock, by context length" in out
    assert "62% words from draft" in out and "top-1 97.0%, KL 0.021" in out and "diverge at 41" in out
    assert "beyond noise" in out and next(tmp_path.glob("e2e-*levers*.json"))
