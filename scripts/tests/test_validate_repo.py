"""Protect public-file discovery and per-topic process isolation."""

from __future__ import annotations

import importlib.util
import ast
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
    monkeypatch.setattr(runner, "_run", lambda command, **options: calls.append(command))
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


def test_scope_matches_a_public_file_or_directory_boundary():
    files = {"topic/tests/test_core.py", "topic-other/test_other.py"}
    assert runner.select_scope(files, "topic") == ["topic/tests/test_core.py"]
    assert runner.select_scope(files, "topic\\tests\\test_core.py") == ["topic/tests/test_core.py"]
    assert runner.select_scope(files, ".") == sorted(files)


@pytest.mark.parametrize("scope", ["", "../topic", "/topic", "D:\\topic", ".local/private", "topic-missing"])
def test_scope_rejects_private_missing_or_escaping_paths(scope):
    with pytest.raises(runner.ValidationError):
        runner.select_scope({"topic/test_core.py"}, scope)


def test_focused_tests_report_scope_and_exclude_only_marked_rendering(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runner, "publication_files", lambda: {"topic/test_core.py", "other/test_other.py"})
    monkeypatch.setattr(runner, "_run", lambda command, **options: calls.append((command, options)))
    runner.run_tests(scope="topic", exclude_rendering=True, threads=2)
    assert calls == [([sys.executable, "-m", "pytest", "-q", "topic/test_core.py", "-m", "not rendering"], {"threads": 2})]
    output = capsys.readouterr().out
    assert "partial" in output and "marked rendering" in output


@pytest.mark.parametrize("threads", [1, 2])
def test_resource_limits_apply_only_to_the_child(monkeypatch, threads):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setenv("UNRELATED_SETTING", "keep")
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **options: calls.append(options) or SimpleNamespace(returncode=0))
    runner._run([sys.executable, "-V"], threads=threads)
    child = calls[0]["env"]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert child[name] == str(threads)
    assert child["MPLBACKEND"] == "Agg"
    assert child["UNRELATED_SETTING"] == "keep"
    assert runner.os.environ["OPENBLAS_NUM_THREADS"] == "8"


def test_apps_scope_keeps_registered_entries_and_isolation(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "STREAMLIT_APPS", (Path("topic/app.py"), Path("other/app.py")))
    monkeypatch.setattr(runner, "_run", lambda command, **options: calls.append((command, options)))
    runner.check_streamlit_apps(scope="topic", threads=2)
    assert calls == [([sys.executable, str(Path(runner.__file__).resolve()), "_streamlit_app", "topic/app.py"], {"threads": 2})]


def test_cli_accepts_explicit_focused_checks(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate_repo.py", "tests", "--scope", "topic", "--exclude-rendering", "--threads", "2"])
    args = runner.parse_arguments()
    assert args.scope == "topic" and args.exclude_rendering and args.threads == 2


@pytest.mark.parametrize("arguments", [
    ["all", "--scope", "topic"],
    ["all", "--exclude-rendering"],
    ["tests", "--exclude-rendering"],
    ["links", "--scope", "topic"],
    ["tests", "ignored-positional-path"],
    ["tests", "--threads", "0"],
    ["tests", "--threads", "-1"],
    ["apps", "--scope", "topic", "--exclude-rendering"],
])
def test_cli_rejects_ambiguous_or_weakened_complete_checks(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["validate_repo.py", *arguments])
    with pytest.raises(SystemExit) as error:
        runner.parse_arguments()
    assert error.value.code == 2


def test_complete_check_keeps_every_stage_and_resource_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "check_syntax", lambda: calls.append("syntax"))
    monkeypatch.setattr(runner, "check_links", lambda: calls.append("links"))
    monkeypatch.setattr(runner, "run_tests", lambda **options: calls.append(("tests", options)))
    monkeypatch.setattr(runner, "check_streamlit_apps", lambda **options: calls.append(("apps", options)))
    runner.run_all(threads=2)
    assert calls == ["syntax", "links", ("tests", {"threads": 2}), ("apps", {"threads": 2})]


@pytest.mark.parametrize("path,name", [
    ("02-unsupervised-recommender-systems/37-dbscan-density-based-clustering/tests/test_visual_dbscan.py", "test_cli_list_and_single_png_export"),
    ("02-unsupervised-recommender-systems/37-dbscan-density-based-clustering/tests/test_visual_dbscan.py", "test_pillow_export_uses_real_epsilon_frames"),
    ("02-unsupervised-recommender-systems/37-dbscan-density-based-clustering/tests/test_visual_dbscan.py", "test_three_feature_geometry_tooltips_and_standalone_export"),
    ("02-unsupervised-recommender-systems/38-hierarchical-clustering/tests/test_visual_hierarchical.py", "test_cli_fast_render_html_and_skip_does_not_overwrite"),
    ("02-unsupervised-recommender-systems/38-hierarchical-clustering/tests/test_visual_hierarchical.py", "test_single_merge_animation_and_static_export_to_temporary_directory"),
    ("02-unsupervised-recommender-systems/40-t-sne-umap/tests/test_visual_lab.py", "test_temporary_gif_png_and_offline_html_exports"),
    ("03-statistics-experimentation/48-multiple-testing/tests/test_visual_lab.py", "test_small_all_export_smoke_and_real_gif_frames"),
])
def test_known_real_exports_are_explicitly_classified(path, name):
    tree = ast.parse((runner.REPOSITORY_ROOT / path).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    assert "pytest.mark.rendering" in [ast.unparse(node) for node in function.decorator_list]
