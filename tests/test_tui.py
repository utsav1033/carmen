import asyncio

import pytest

pytest.importorskip("textual")

from carmen.tui import Carmen, HomeScreen, logo, plain_failure  # noqa: E402


def test_logo_is_six_rows_of_block_type():
    assert str(logo()).count("\n") == 5 and "█" in str(logo())


def test_plain_failure_names_the_bug():
    v = {"stage": "correctness", "configs": [{"failures": [
        {"patterns": ["NaN where a number was expected"], "where": "only the last elements of rows: tail handling",
         "smallest_failing": "rows=1, n=33"}]}]}
    assert plain_failure(v) == "tail bug, fails at rows=1, n=33"


def test_app_opens_on_home_with_every_op(tmp_path):
    async def go():
        app = Carmen(runs_dir=tmp_path / "runs", memory_dir=tmp_path / "mem")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)
            assert app.screen.query_one("#ops").option_count == 2
    asyncio.run(go())
