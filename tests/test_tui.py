import asyncio

import pytest

pytest.importorskip("textual")

from carmen.tui import Carmen, logo  # noqa: E402


def test_logo_is_six_rows_of_block_type():
    assert str(logo()).count("\n") == 5 and "█" in str(logo())


def test_app_mounts_and_shows_a_golden_kernel(tmp_path):
    async def go():
        app = Carmen(runs_dir=tmp_path / "runs", memory_dir=tmp_path / "mem")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            assert app.current.op == "masked_softmax"
            assert "threadgroup_barrier" in app.current.kernel.source
            await pilot.press("f5")
    asyncio.run(go())
