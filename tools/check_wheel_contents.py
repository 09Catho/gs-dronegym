"""Verify that a built wheel ships every tracked package module.

Release 0.3.0 reached PyPI without ``gs_dronegym/env`` because an over-broad
ignore rule kept those files out of version control. A build from a clean
checkout silently omitted them, and ``import gs_dronegym`` failed for every
user, while the test suite passed because it ran from a working tree that still
had the files. This check compares a wheel against a clean source tree so that
class of release cannot recur.

Usage:
    python tools/check_wheel_contents.py dist/gs_dronegym-*.whl --source-root .
"""

from __future__ import annotations

import argparse
import configparser
import sys
import zipfile
from pathlib import Path

PACKAGE = "gs_dronegym"


def _load_pyproject(source_root: Path) -> dict[str, object]:
    """Parse the project's ``pyproject.toml``.

    Args:
        source_root: Root of a clean source tree.

    Returns:
        Parsed TOML document.

    Raises:
        SystemExit: If the interpreter lacks ``tomllib``.
    """
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - Python 3.10 only
        raise SystemExit("check_wheel_contents.py requires Python 3.11 or newer.") from exc
    with (source_root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _source_modules(source_root: Path) -> set[str]:
    """List package modules present in a source tree.

    Args:
        source_root: Root of a clean source tree.

    Returns:
        POSIX-style paths of every module under the package.
    """
    return {
        path.relative_to(source_root).as_posix()
        for path in (source_root / PACKAGE).rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _dist_info_member(entries: set[str], filename: str) -> str | None:
    """Find a file inside the wheel's ``.dist-info`` directory.

    Args:
        entries: All archive member names.
        filename: File to look for.

    Returns:
        Archive member name, or ``None`` if absent.
    """
    for entry in entries:
        parts = entry.split("/")
        if len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] == filename:
            return entry
    return None


def _module_path_candidates(target: str) -> tuple[str, str]:
    """Return archive paths that could implement an entry-point target.

    Args:
        target: Entry-point target such as ``package.module:function``.

    Returns:
        The module-file and package-init candidate paths.
    """
    module = target.split(":", 1)[0].strip().replace(".", "/")
    return f"{module}.py", f"{module}/__init__.py"


def check_wheel(wheel: Path, source_root: Path) -> list[str]:
    """Compare a wheel against a source tree and collect every discrepancy.

    Args:
        wheel: Path to the built wheel.
        source_root: Root of a clean source tree.

    Returns:
        Human-readable problems; empty when the wheel is complete.
    """
    problems: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        entries = set(archive.namelist())
        pyproject = _load_pyproject(source_root)
        project = pyproject["project"]
        assert isinstance(project, dict)

        missing = sorted(_source_modules(source_root) - entries)
        problems.extend(f"module missing from wheel: {path}" for path in missing)

        top_level = {entry.split("/", 1)[0] for entry in entries}
        unexpected = sorted(
            name for name in top_level if name != PACKAGE and not name.endswith(".dist-info")
        )
        problems.extend(f"unexpected top-level entry in wheel: {name}" for name in unexpected)

        metadata_member = _dist_info_member(entries, "METADATA")
        if metadata_member is None:
            problems.append("wheel has no METADATA file")
        else:
            metadata = archive.read(metadata_member).decode("utf-8")
            versions = [
                line.split(":", 1)[1].strip()
                for line in metadata.splitlines()
                if line.startswith("Version:")
            ]
            if versions != [str(project["version"])]:
                problems.append(
                    f"wheel version {versions} does not match pyproject {project['version']}"
                )

        expected_scripts = dict(project.get("scripts", {}))
        entry_points_member = _dist_info_member(entries, "entry_points.txt")
        installed_scripts: dict[str, str] = {}
        if entry_points_member is not None:
            parser = configparser.ConfigParser()
            parser.optionxform = str  # type: ignore[assignment,method-assign]
            parser.read_string(archive.read(entry_points_member).decode("utf-8"))
            if parser.has_section("console_scripts"):
                installed_scripts = dict(parser.items("console_scripts"))

        for name in sorted(set(expected_scripts) - set(installed_scripts)):
            problems.append(f"console script missing from wheel: {name}")
        for name in sorted(set(installed_scripts) - set(expected_scripts)):
            problems.append(f"console script not declared in pyproject: {name}")
        for name, target in sorted(expected_scripts.items()):
            if name in installed_scripts and installed_scripts[name].strip() != str(target):
                problems.append(f"console script {name} targets {installed_scripts[name]}")
            if not any(path in entries for path in _module_path_candidates(str(target))):
                problems.append(f"console script {name} target module absent: {target}")

        print(
            f"{wheel.name}: {len(entries)} entries, {len(_source_modules(source_root))} "
            f"source modules, {len(installed_scripts)} console scripts"
        )
    return problems


def main() -> int:
    """Run the wheel contents check.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheel", type=Path, help="Built wheel to inspect.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("."),
        help="Root of a clean source tree to compare against.",
    )
    args = parser.parse_args()
    problems = check_wheel(args.wheel, args.source_root)
    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        return 1
    print("PASS: wheel contains every tracked module, declared script and matching version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
