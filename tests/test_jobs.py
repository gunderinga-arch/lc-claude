from __future__ import annotations

import asyncio
import logging

import pytest

from bot.services.jobs import JobBusyError, JobManager, ProgressNotifier


async def _sleep_forever() -> None:
    await asyncio.sleep(100)


async def _quick(events: list[str]) -> None:
    events.append("ran")


async def _boom() -> None:
    raise ValueError("boom")


async def test_submit_runs_job_and_registry_empties() -> None:
    manager = JobManager()
    events: list[str] = []
    task = manager.submit(1, _quick(events))
    await task
    assert events == ["ran"]
    assert manager.is_running(1) is False


async def test_second_submit_while_running_raises_and_closes_coro() -> None:
    manager = JobManager()
    manager.submit(1, _sleep_forever())
    rejected = _sleep_forever()
    with pytest.raises(JobBusyError):
        manager.submit(1, rejected)
    assert rejected.cr_frame is None  # closed, not left dangling
    manager.cancel(1)


async def test_cancel_returns_true_then_false() -> None:
    manager = JobManager()
    manager.submit(1, _sleep_forever())
    assert manager.cancel(1) is True
    await asyncio.sleep(0)
    assert manager.cancel(1) is False


async def test_new_job_can_be_submitted_right_after_previous_finished() -> None:
    manager = JobManager()
    events: list[str] = []
    await manager.submit(1, _quick(events))
    task = manager.submit(1, _quick(events))
    await task
    assert events == ["ran", "ran"]


async def test_failing_job_is_logged_and_removed(caplog: pytest.LogCaptureFixture) -> None:
    manager = JobManager()
    with caplog.at_level(logging.ERROR, logger="bot.services.jobs"):
        task = manager.submit(1, _boom())
        with pytest.raises(ValueError, match="boom"):
            await task
        await asyncio.sleep(0)
    assert manager.is_running(1) is False
    assert any("failed" in record.message for record in caplog.records)
    assert any(record.exc_info is not None for record in caplog.records)


async def test_shutdown_cancels_running_jobs_and_blocks_submit() -> None:
    manager = JobManager()
    task = manager.submit(1, _sleep_forever())
    await manager.shutdown(timeout_s=1.0)
    assert task.cancelled()
    with pytest.raises(RuntimeError):
        manager.submit(1, _sleep_forever())


async def test_progress_notifier_calls_send_immediately_and_repeatedly() -> None:
    calls: list[int] = []

    async def send() -> None:
        calls.append(1)

    async with ProgressNotifier(send, interval_s=0.01):
        assert calls == [1]
        await asyncio.sleep(0.05)
    assert len(calls) >= 3


async def test_progress_notifier_stops_after_exit() -> None:
    calls: list[int] = []

    async def send() -> None:
        calls.append(1)

    async with ProgressNotifier(send, interval_s=0.01):
        await asyncio.sleep(0.03)
    count_after_exit = len(calls)
    await asyncio.sleep(0.05)
    assert len(calls) == count_after_exit


async def test_progress_notifier_survives_send_that_raises(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[int] = []

    async def send() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("send failed")

    with caplog.at_level(logging.WARNING, logger="bot.services.jobs"):
        async with ProgressNotifier(send, interval_s=0.01):
            await asyncio.sleep(0.05)
    assert len(calls) >= 3
    assert any(record.levelno == logging.WARNING for record in caplog.records)


async def test_progress_notifier_does_not_swallow_body_exception() -> None:
    async def send() -> None:
        pass

    with pytest.raises(ValueError, match="body error"):
        async with ProgressNotifier(send, interval_s=0.01):
            raise ValueError("body error")


async def test_progress_notifier_body_cancellation_propagates() -> None:
    async def send() -> None:
        pass

    async def run() -> None:
        async with ProgressNotifier(send, interval_s=0.01):
            await asyncio.sleep(100)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
