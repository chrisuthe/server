"""Tests for Sendspin DSP output chunk coalescing helpers."""

from __future__ import annotations

import asyncio
from collections import deque

import pytest

from music_assistant.providers.sendspin.player import _DSPChunkRequest, _resolve_ready_dsp_requests


@pytest.mark.asyncio
async def test_resolve_ready_dsp_requests_spans_multiple_requests() -> None:
    """Output bytes should be distributed across pending requests in FIFO order."""
    loop = asyncio.get_running_loop()
    first = loop.create_future()
    second = loop.create_future()
    pending = deque(
        [
            _DSPChunkRequest(first, expected_bytes=6, buffer=bytearray()),
            _DSPChunkRequest(second, expected_bytes=4, buffer=bytearray()),
        ]
    )

    output_buffer = bytearray(b"abcdefgh")
    _resolve_ready_dsp_requests(output_buffer, pending)

    assert first.done()
    assert first.result() == b"abcdef"
    assert not second.done()
    assert pending[0].buffer == b"gh"
    assert output_buffer == b""

    output_buffer.extend(b"ij")
    _resolve_ready_dsp_requests(output_buffer, pending)

    assert second.done()
    assert second.result() == b"ghij"
    assert not pending
    assert output_buffer == b""


@pytest.mark.asyncio
async def test_resolve_ready_dsp_requests_keeps_partial_buffer() -> None:
    """Request should remain pending until expected byte count is reached."""
    loop = asyncio.get_running_loop()
    request_future = loop.create_future()
    pending = deque(
        [
            _DSPChunkRequest(
                request_future,
                expected_bytes=5,
                buffer=bytearray(b"ab"),
            )
        ]
    )

    output_buffer = bytearray(b"cd")
    _resolve_ready_dsp_requests(output_buffer, pending)

    assert not request_future.done()
    assert pending[0].buffer == b"abcd"
    assert output_buffer == b""

    output_buffer.extend(b"e")
    _resolve_ready_dsp_requests(output_buffer, pending)

    assert request_future.done()
    assert request_future.result() == b"abcde"
    assert not pending
    assert output_buffer == b""
