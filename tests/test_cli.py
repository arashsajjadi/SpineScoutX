"""Tests for the SpineScoutX CLI entrypoint."""

from __future__ import annotations

import pytest

from spinescoutx.cli import main

_SUBCOMMANDS: tuple[str, ...] = (
    "doctor",
    "prepare-rsna",
    "prepare-spider",
    "train-classifier",
    "train-segmenter",
    "train-anatomy-guided",
    "evaluate",
    "ablate",
    "report",
    "figure",
    "benchmark",
)


def test_top_level_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("name", _SUBCOMMANDS)
def test_subcommand_help_exits_zero(name: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main([name, "--help"])
    assert exc.value.code == 0


def test_doctor_returns_zero() -> None:
    assert main(["doctor"]) == 0


def test_doctor_json_returns_zero() -> None:
    assert main(["doctor", "--json"]) == 0
