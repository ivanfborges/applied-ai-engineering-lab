"""Protect public-file discovery and per-topic process isolation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "repo_validation_under_test", Path(__file__).resolve().parents[1] / "validate_repo.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_publication_files_uses_git_public_candidates(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=b"README.md\0topic\\tests\\test_core.py\0")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.publication_files() == {"README.md", "topic/tests/test_core.py"}
    command, options = calls[0]
    assert command == ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    assert options["cwd"] == runner.REPOSITORY_ROOT
    assert options["check"] is True


def test_each_public_test_file_runs_in_a_fresh_process(monkeypatch):
    files = {
        "01-classical-machine-learning/35-topic/tests/test_core.py",
        "00-foundations/02-topic/tests/test_core.py",
        "scripts/tests/test_validate_repo.py",
        "topic/example.py",
        "topic/test_notes.md",
    }
    calls = []
    monkeypatch.setattr(runner, "publication_files", lambda: files)
    monkeypatch.setattr(runner, "_run", calls.append)
    runner.run_tests()
    expected = sorted(path for path in files if path.endswith("test_core.py") or path.startswith("scripts/"))
    assert calls == [[sys.executable, "-m", "pytest", "-q", path] for path in expected]
    assert all(len(command) == 5 for command in calls)


def test_no_discovered_tests_is_an_error(monkeypatch):
    monkeypatch.setattr(runner, "publication_files", lambda: {"README.md", "example.py"})
    with pytest.raises(runner.ValidationError, match="No public test files"):
        runner.run_tests()


def test_nonzero_subprocess_stops_validation(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=2)
    )
    with pytest.raises(runner.ValidationError, match="exit code 2"):
        runner._run([sys.executable, "-m", "pytest", "-q", "test_core.py"])


@pytest.mark.parametrize("target", ["#local-heading", "https://example.org/doc", "mailto:a@example.org", "//example.org/doc"])
def test_nonlocal_or_heading_links_are_outside_path_check(target):
    assert runner._resolve_publication_target("docs/guide.md", target) is None


@pytest.mark.parametrize("target,expected", [
    ("../README.md#start", "README.md"),
    ("../topic/", "topic"),
    ("../docs/file%20name.md", "docs/file name.md"),
    ("/README.md", "README.md"),
    ("..\\README.md", "README.md"),
])
def test_local_links_resolve_relative_to_source(target, expected):
    assert runner._resolve_publication_target("docs/guide.md", target) == expected


@pytest.mark.parametrize("target", ["../../private.md", "D:/local/notes.md", "C:\\local\\notes.md"])
def test_unpublishable_paths_fail(target):
    with pytest.raises(runner.ValidationError):
        runner._resolve_publication_target("docs/guide.md", target)


def test_link_to_existing_but_ignored_file_is_rejected(monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("[private](.local/notes.md)\n", encoding="utf-8")
    (tmp_path / ".local").mkdir()
    (tmp_path / ".local/notes.md").write_text("Local only\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "publication_files", lambda: {"README.md"})
    with pytest.raises(runner.ValidationError, match="Broken, private, or untracked"):
        runner.check_links()


def test_directory_link_with_public_content_is_accepted(monkeypatch, tmp_path):
    (tmp_path / "README.md").write_text("[topic](topic/)\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(runner, "publication_files", lambda: {"README.md", "topic/example.py"})
    runner.check_links()
