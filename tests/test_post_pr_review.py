"""Tests for the Automated PR Review posting script."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / ".github" / "scripts" / "post_pr_review.py"

FINDING = {
    "severity": "PROBLEM",
    "path": "music_assistant/helpers/util.py",
    "line": 42,
    "issue": "Swallows the exception without re-raising.",
    "principle": "Never swallow errors",
    "citation_url": "https://example.invalid/standards#errors",
    "suggestion": None,
    "scaffold": None,
}
# What run 31018782625 actually left in findings.json: prose from a failed CLI run, no JSON.
POLICY_ERROR = "Error: Access denied by policy settings\nRequest ID: F02D:2F23E3:73CEC5F\n"


@pytest.fixture
def post_pr_review() -> types.ModuleType:
    """Load the workflow script, which lives under ``.github`` and is not importable."""
    spec = importlib.util.spec_from_file_location("post_pr_review", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def posted(
    post_pr_review: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[str], str | None]]:
    """Record every ``gh`` call ``main`` makes instead of reaching GitHub."""
    calls: list[tuple[list[str], str | None]] = []

    def fake_gh(args: list[str], inp: str | None = None) -> str:
        calls.append((args, inp))
        return ""

    monkeypatch.setattr(post_pr_review, "gh", fake_gh)
    monkeypatch.setenv("REPO", "music-assistant/server")
    monkeypatch.setenv("PR_NUMBER", "5357")
    return calls


def run_main(
    module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw: str,
) -> None:
    """Run the script's ``main`` against a findings file holding ``raw``."""
    path = tmp_path / "findings.json"
    path.write_text(raw, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["post_pr_review.py", str(path)])
    module.main()


def review_payload(calls: list[tuple[list[str], str | None]]) -> dict[str, Any]:
    """Return the decoded payload of the single review that was posted."""
    assert len(calls) == 1
    args, inp = calls[0]
    assert args[1] == "repos/music-assistant/server/pulls/5357/reviews"
    assert inp is not None
    payload: dict[str, Any] = json.loads(inp)
    return payload


def test_findings_are_extracted_from_an_array_followed_by_prose(
    post_pr_review: types.ModuleType,
) -> None:
    """A valid array is found even when the model adds trailing commentary."""
    raw = f"{json.dumps([FINDING])}\nThat is all I found.\n"
    assert post_pr_review.extract_findings(raw) == [FINDING]


def test_an_empty_array_is_a_real_result(post_pr_review: types.ModuleType) -> None:
    """A genuinely empty array parses as an empty list rather than raising."""
    assert post_pr_review.extract_findings("[]\n") == []


def test_a_stray_empty_array_in_prose_does_not_mask_later_findings(
    post_pr_review: types.ModuleType,
) -> None:
    """Findings printed after an empty array elsewhere in the output are still found."""
    raw = f"Nothing in [] so far, then:\n{json.dumps([FINDING])}"
    assert post_pr_review.extract_findings(raw) == [FINDING]


@pytest.mark.parametrize(
    "raw",
    [
        POLICY_ERROR,
        "",
        '[{"severity": "PROBLEM", "path": "a.py"',  # truncated array
        "[1, 2, 3]",  # a list, but not of findings
    ],
    ids=["error_prose", "empty_file", "truncated_array", "not_findings"],
)
def test_output_without_a_findings_array_raises(
    post_pr_review: types.ModuleType,
    raw: str,
) -> None:
    """Anything that is not a findings array is an error, not an empty review."""
    with pytest.raises(ValueError, match="no parseable JSON array"):
        post_pr_review.extract_findings(raw)


def test_unparseable_output_fails_the_run_and_posts_nothing(
    post_pr_review: types.ModuleType,
    posted: list[tuple[list[str], str | None]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed review run exits non-zero, annotates the log, and posts nothing."""
    with pytest.raises(SystemExit) as excinfo:
        run_main(post_pr_review, monkeypatch, tmp_path, POLICY_ERROR)

    assert excinfo.value.code == 1
    assert posted == []
    stderr = capsys.readouterr().err
    assert "::error title=Automated PR Review produced no parseable findings::" in stderr
    assert "Access denied by policy settings%0ARequest ID" in stderr


def test_an_empty_findings_array_posts_the_clean_review(
    post_pr_review: types.ModuleType,
    posted: list[tuple[list[str], str | None]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A review that ran and found nothing still reports a clean result."""
    run_main(post_pr_review, monkeypatch, tmp_path, "[]\n")

    payload = review_payload(posted)
    assert "No standards violations found." in payload["body"]
    assert "comments" not in payload


def test_findings_are_posted_inline(
    post_pr_review: types.ModuleType,
    posted: list[tuple[list[str], str | None]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A finding anchored to a diff line is posted as an inline review comment."""
    run_main(post_pr_review, monkeypatch, tmp_path, json.dumps([FINDING]))

    payload = review_payload(posted)
    assert "**1 finding(s)**" in payload["body"]
    assert payload["comments"] == [
        {
            "path": FINDING["path"],
            "line": FINDING["line"],
            "side": "RIGHT",
            "body": (
                "**[PROBLEM]** Swallows the exception without re-raising.\n\n"
                f"_Standard: {FINDING['principle']} — {FINDING['citation_url']}_"
            ),
        },
    ]


def test_a_finding_without_a_line_becomes_an_unanchored_note(
    post_pr_review: types.ModuleType,
    posted: list[tuple[list[str], str | None]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A finding that cannot be anchored to a diff line is listed in the review body instead."""
    run_main(post_pr_review, monkeypatch, tmp_path, json.dumps([{**FINDING, "line": None}]))

    payload = review_payload(posted)
    assert "### Notes (not anchored to diff lines)" in payload["body"]
    assert "comments" not in payload


def test_a_closed_pr_falls_back_to_a_plain_comment(
    post_pr_review: types.ModuleType,
    posted: list[tuple[list[str], str | None]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With the reviews API refusing, the findings still land as a plain PR comment."""

    def refuse_reviews(args: list[str], inp: str | None = None) -> str:
        posted.append((args, inp))
        if "reviews" in args[1]:
            raise RuntimeError("Pull request review cannot be submitted")
        return ""

    monkeypatch.setattr(post_pr_review, "gh", refuse_reviews)
    run_main(post_pr_review, monkeypatch, tmp_path, json.dumps([FINDING]))

    endpoints = [args[1] for args, _ in posted]
    assert endpoints == [
        "repos/music-assistant/server/pulls/5357/reviews",  # inline
        "repos/music-assistant/server/pulls/5357/reviews",  # summary-only
        "repos/music-assistant/server/issues/5357/comments",  # plain comment
    ]
    assert "Swallows the exception" in str(posted[-1][1])


def test_annotation_escapes_percent_before_newlines(post_pr_review: types.ModuleType) -> None:
    """Escaping ``%`` first keeps the newline escapes themselves intact."""
    assert post_pr_review._annotation_safe("50% done\nthen failed") == "50%25 done%0Athen failed"


def test_annotation_truncates_and_names_empty_output(post_pr_review: types.ModuleType) -> None:
    """Long output is cut to the limit and marked as cut; no output at all is stated explicitly."""
    excerpt = post_pr_review._annotation_safe("x" * 5000)
    assert excerpt == "x" * post_pr_review.EXCERPT_LIMIT + "…"
    assert post_pr_review._annotation_safe("x" * post_pr_review.EXCERPT_LIMIT).endswith("x")
    assert post_pr_review._annotation_safe("   \n") == "(nothing)"
