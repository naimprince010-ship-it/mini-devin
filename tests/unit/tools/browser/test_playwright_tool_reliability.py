import asyncio

from mini_devin.tools.browser.playwright_tool import PlaywrightBrowserTool


def test_retry_settings_clamped_bounds() -> None:
    tool = PlaywrightBrowserTool()

    retries, delay = tool._retry_settings({"retries": 99, "retry_delay_ms": 999999})
    assert retries == 5
    assert delay == 5000

    retries, delay = tool._retry_settings({"retries": -4, "retry_delay_ms": -100})
    assert retries == 0
    assert delay == 0


def test_retry_action_eventual_success() -> None:
    tool = PlaywrightBrowserTool()
    state = {"attempts": 0}

    async def flaky_op() -> str:
        state["attempts"] += 1
        if state["attempts"] < 3:
            raise RuntimeError("temporary failure")
        return "ok"

    result = asyncio.run(tool._retry_action("click", flaky_op, retries=2, delay_ms=0))

    assert result == "ok"
    assert state["attempts"] == 3


def test_retry_action_raises_after_exhausted_retries() -> None:
    tool = PlaywrightBrowserTool()

    async def failing_op() -> None:
        raise RuntimeError("still failing")

    try:
        asyncio.run(tool._retry_action("type", failing_op, retries=1, delay_ms=0))
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "failed after 2 attempt" in str(exc)

    assert raised
