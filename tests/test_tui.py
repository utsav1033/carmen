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
            from carmen import ops
            ol = app.screen.query_one("#ops")
            ids = [ol.get_option_at_index(i).id for i in range(ol.option_count)]
            assert set(i for i in ids if i) == set(ops.OPS) | {f"wip:{w['name']}" for w in ops.WIP}
            from rich.console import Console
            from carmen.tui import proven_text
            c = Console(width=160, record=True)
            c.print(proven_text())
            text = c.export_text()
            assert "1.03×" in text and "75/75" in text
            assert app.screen.query_one("#models").option_count >= 1
            assert "mlp_up" in str(app.screen.query_one("#about").render())
    asyncio.run(go())


def test_wip_rows_do_not_start_a_cook(tmp_path):
    async def go():
        from carmen.tui import HomeScreen
        app = Carmen(runs_dir=tmp_path / "runs", memory_dir=tmp_path / "mem")
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            ol = app.screen.query_one("#ops")
            ol.highlighted = ol.option_count - 1
            await pilot.pause()
            assert "work in progress" in str(app.screen.query_one("#about").render())
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)
    asyncio.run(go())
