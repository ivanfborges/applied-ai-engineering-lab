"""Run repository-wide checks locally and in continuous integration."""

from __future__ import annotations

import argparse
import ast
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APPS = (
    Path('00-foundations/12-hypothesis-testing/hypothesis_testing_visual_lab.py'),
    Path("00-foundations/02-linear-algebra-vectors-matrices/visualizations/app.py"),
    Path(
        "00-foundations/04-eigenvalues-eigenvectors-pca-svd/"
        "visual_lab/interactive_lab.py"
    ),
    Path("00-foundations/07-probability-essentials/visual_lab.py"),
    Path("00-foundations/08-probability-distributions/interactive_dashboard.py"),
)
INLINE_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK_PATTERN = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
HTML_LINK_PATTERN = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


class ValidationError(RuntimeError):
    """Represent an expected validation failure with a concise message."""


def publication_files() -> set[str]:
    """Return tracked and non-ignored untracked files relative to the root."""
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ]
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return {
        path.decode("utf-8").replace("\\", "/")
        for path in result.stdout.split(b"\0")
        if path
    }


def check_syntax() -> None:
    """Parse every public Python source without creating bytecode files."""
    python_files = sorted(
        path for path in publication_files() if path.lower().endswith(".py")
    )
    failures: list[str] = []
    for relative_path in python_files:
        path = REPOSITORY_ROOT / relative_path
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        except (SyntaxError, UnicodeDecodeError) as error:
            failures.append(f"{relative_path}: {error}")

    if failures:
        raise ValidationError("Python syntax failures:\n" + "\n".join(failures))
    print(f"Syntax check passed for {len(python_files)} Python files.")


def _normalize_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(' "', 1)[0].split(" '", 1)[0].strip()


def _resolve_publication_target(source: str, target: str) -> str | None:
    if not target or target.startswith("#"):
        return None
    if WINDOWS_ABSOLUTE_PATTERN.match(target):
        raise ValidationError(f"{source}: local absolute path is not publishable: {target}")

    parsed = urlsplit(target)
    if parsed.scheme or target.startswith("//"):
        return None

    path_part = unquote(parsed.path).replace("\\", "/")
    if not path_part:
        return None
    if path_part.startswith("/"):
        resolved = posixpath.normpath(path_part.lstrip("/"))
    else:
        resolved = posixpath.normpath(
            posixpath.join(posixpath.dirname(source), path_part)
        )
    if resolved == ".." or resolved.startswith("../"):
        raise ValidationError(f"{source}: link escapes the repository: {target}")
    return resolved


def check_links() -> None:
    """Verify that local Markdown and HTML links target public files."""
    files = publication_files()
    markdown_files = sorted(path for path in files if path.lower().endswith(".md"))
    failures: list[str] = []
    checked = 0

    for source in markdown_files:
        text = (REPOSITORY_ROOT / source).read_text(encoding="utf-8")
        raw_targets = (
            INLINE_LINK_PATTERN.findall(text)
            + REFERENCE_LINK_PATTERN.findall(text)
            + HTML_LINK_PATTERN.findall(text)
        )
        for raw_target in raw_targets:
            target = _normalize_link_target(raw_target)
            try:
                resolved = _resolve_publication_target(source, target)
            except ValidationError as error:
                failures.append(str(error))
                continue
            if resolved is None:
                continue

            checked += 1
            directory_prefix = resolved.rstrip("/") + "/"
            if resolved not in files and not any(
                path.startswith(directory_prefix) for path in files
            ):
                failures.append(f"{source}: {target} -> {resolved}")

    if failures:
        raise ValidationError(
            "Broken, private, or untracked local links:\n" + "\n".join(failures)
        )
    print(f"Internal link check passed for {checked} links in {len(markdown_files)} files.")


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if result.returncode:
        raise ValidationError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def run_tests() -> None:
    """Run every public test file in isolation to avoid topic import clashes."""
    test_files = sorted(
        path
        for path in publication_files()
        if Path(path).name.startswith("test_") and path.lower().endswith(".py")
    )
    if not test_files:
        raise ValidationError("No public test files were discovered.")

    for relative_path in test_files:
        print(f"Running tests: {relative_path}", flush=True)
        _run([sys.executable, "-m", "pytest", "-q", relative_path])
    print(f"Test check passed for {len(test_files)} isolated test files.")


def _run_streamlit_app(relative_path: str) -> None:
    """Render one Streamlit app with AppTest in an isolated process."""
    from streamlit.testing.v1 import AppTest

    app_path = (REPOSITORY_ROOT / relative_path).resolve()
    if not app_path.is_file():
        raise ValidationError(f"Streamlit app does not exist: {relative_path}")

    sys.path.insert(0, str(app_path.parent))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    app = AppTest.from_file(str(app_path), default_timeout=45)
    app.run(timeout=45)
    exceptions = [str(item.value) for item in app.exception]
    if exceptions:
        raise ValidationError(
            f"Streamlit smoke test failed for {relative_path}:\n"
            + "\n".join(exceptions)
        )
    print(f"Streamlit smoke test passed: {relative_path}")


def check_streamlit_apps() -> None:
    """Smoke-test selected apps in separate processes to isolate imports."""
    for app_path in STREAMLIT_APPS:
        _run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_streamlit_app",
                app_path.as_posix(),
            ]
        )


def run_all() -> None:
    """Run the complete local equivalent of the CI workflow."""
    check_syntax()
    check_links()
    run_tests()
    check_streamlit_apps()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("all", "syntax", "links", "tests", "apps", "_streamlit_app"),
    )
    parser.add_argument("path", nargs="?")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        if args.command == "all":
            run_all()
        elif args.command == "syntax":
            check_syntax()
        elif args.command == "links":
            check_links()
        elif args.command == "tests":
            run_tests()
        elif args.command == "apps":
            check_streamlit_apps()
        elif args.command == "_streamlit_app":
            if not args.path:
                raise ValidationError("_streamlit_app requires a repository path.")
            _run_streamlit_app(args.path)
    except (OSError, subprocess.SubprocessError, ValidationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
