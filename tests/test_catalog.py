import json

from carmen import catalog, target


def test_find_knows_aliases_ids_and_unknown_models():
    assert catalog.find("qwen3b").id == "mlx-community/Qwen2.5-3B-Instruct-4bit"
    assert catalog.find("mlx-community/Llama-3.2-3B-Instruct-4bit").alias == "llama3b"
    other = catalog.find("someone/new-model-4bit")
    assert other.id == "someone/new-model-4bit" and other.gb == 0.0 and other.label == "new-model-4bit"


def test_every_catalog_model_is_a_cli_alias_too():
    for m in catalog.MODELS:
        assert target.ALIASES.get(m.alias) == m.id


def test_fits_leaves_room_for_the_system_and_counts_copies():
    assert catalog.fits(1.74, ram=16) == "yes"
    assert catalog.fits(4.28, copies=3, ram=16) == "no"  # speedup loads three 7B copies: it would swap
    assert catalog.fits(4.28, copies=2, ram=16) == "tight"
    assert catalog.fits(0.0, ram=16) == "unknown" and catalog.fits(1.0, ram=0) == "unknown"


def test_downloaded_looks_for_weights_in_the_hf_cache(tmp_path):
    mid = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    assert not catalog.downloaded(mid, tmp_path)
    snap = tmp_path / "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert not catalog.downloaded(mid, tmp_path)  # a config alone is a broken download
    (snap / "model.safetensors").write_text("")
    assert catalog.downloaded(mid, tmp_path)


def _row(mode, turns, **kw):
    return {"mode": mode, "decode_tps": sum(turns) / len(turns), "decode_turns": turns, **kw}


def test_measured_reads_this_macs_fusion_and_levers_runs(tmp_path):
    mid = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    (tmp_path / "e2e-a-20261001.json").write_text(json.dumps({
        "model": mid, "decode_speed_limit_tps": 100.0,
        "results": [_row("stock", [50.0, 50.0, 50.0]), _row("mlp+compile", [47.5, 47.5, 47.5])]}))
    (tmp_path / "e2e-b-20261002.json").write_text(json.dumps({
        "model": mid + "-levers",
        "results": [_row("stock", [40.0, 40.0], context=8192), _row("spec", [60.0, 60.0], context=8192),
                    _row("kv8", [34.0, 34.0], context=8192)]}))
    (tmp_path / "e2e-c-20261003.json").write_text(json.dumps({  # another model: ignored
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "results": [_row("stock", [20.0])]}))
    got = catalog.measured(mid, tmp_path)
    assert round(got["fusion"], 3) == 0.95 and got["limit_share"] == 0.5
    assert got["spec"] == {8192: 1.5} and round(got["kv8"][8192], 2) == 0.85
    assert catalog.summary(mid, tmp_path) == "carmen kernels 0.95× · spec 1.50× @8k · kv8 0.85× @8k"
    assert catalog.measured(mid, tmp_path / "missing") == {}


def test_measured_keeps_quality_and_the_newest_speed_limit(tmp_path):
    import os
    mid = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    old = tmp_path / "e2e-z-old.json"  # sorts last by name, but is older on disk
    old.write_text(json.dumps({"model": mid, "decode_speed_limit_tps": 100.0,
                               "results": [_row("stock", [82.0]), _row("mlp+compile", [80.0])]}))
    os.utime(old, (1, 1))
    (tmp_path / "e2e-a-new.json").write_text(json.dumps({
        "model": mid + "-levers", "peak_gbps": 100.0,
        "results": [_row("stock", [50.0, 50.0], context=512, weight_bytes=int(1.9e9), kv_bytes=int(0.1e9)),
                    _row("kv4", [48.0, 48.0], context=512, top1=0.05)]}))
    got = catalog.measured(mid, tmp_path)
    assert got["quality"]["kv4"][512] == 0.05
    assert round(got["limit_share"], 2) == 1.0  # from the newer levers run: 50 tok/s x 2 GB / 100 GB/s
