"""Smoke-test an installed GS-DroneGym wheel from outside the source tree.

Run this with the interpreter of a clean environment into which only the built
wheel was installed. It refuses to run against a source checkout, because a test
that imports the working tree cannot notice files missing from a release, which
is exactly how 0.3.0 shipped without ``gs_dronegym/env``.

Usage:
    python tools/wheel_smoke_test.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np

DISTRIBUTION = "gs-dronegym"

EXPECTED_SCRIPTS = frozenset(
    {
        "gs-dronegym-build-scene",
        "gs-dronegym-evaluate",
        "gs-dronegym-evaluate-expert",
        "gs-dronegym-generate-dataset",
        "gs-dronegym-inspect-dataset",
        "gs-dronegym-live-view",
        "gs-dronegym-preview-dataset-task",
        "gs-dronegym-train-bc",
        "gs-dronegym-validate-dataset",
    }
)

#: Modules whose absence would break the public API, listed explicitly because a
#: filesystem walk cannot report a subpackage that was never installed.
REQUIRED_MODULES = (
    "gs_dronegym.env.drone_env",
    "gs_dronegym.dynamics.quadrotor",
    "gs_dronegym.renderer.camera_model",
    "gs_dronegym.renderer.gsplat_renderer",
    "gs_dronegym.scene.occupancy",
    "gs_dronegym.scene.synthetic",
    "gs_dronegym.data.generation",
    "gs_dronegym.baselines.behavior_cloning",
    "gs_dronegym.benchmarks.drone",
    "gs_dronegym.utils.instruction_encoder",
    "gs_dronegym.cli._scene",
)


def check_installed_location() -> str:
    """Confirm the package resolves to an installed wheel, not a checkout.

    Returns:
        Installed distribution and package directory.

    Raises:
        AssertionError: If the package was imported from a source tree.
    """
    import gs_dronegym

    location = Path(gs_dronegym.__file__).resolve()
    if "site-packages" not in location.parts:
        raise AssertionError(f"imported from {location}; expected an installed wheel")
    version = importlib.metadata.version(DISTRIBUTION)
    return f"{DISTRIBUTION} {version} at {location.parent}"


def check_required_modules() -> str:
    """Import each module the public API depends on.

    Returns:
        Summary of imported modules.

    Raises:
        AssertionError: If any required module cannot be imported.
    """
    failures = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {exc!r}")
    if failures:
        raise AssertionError("; ".join(failures))
    return f"{len(REQUIRED_MODULES)} required modules"


def check_every_module_imports() -> str:
    """Import every module shipped in the package.

    ``pkgutil.walk_packages`` silently ignores import errors unless given an
    error handler, which would hide a broken subpackage, so one is supplied.

    Returns:
        Number of modules imported.

    Raises:
        AssertionError: If any module fails to import.
    """
    import gs_dronegym

    failures: list[str] = []

    def record_failure(name: str) -> None:
        failures.append(f"{name}: {sys.exc_info()[1]!r}")

    count = 0
    for module in pkgutil.walk_packages(
        gs_dronegym.__path__, prefix="gs_dronegym.", onerror=record_failure
    ):
        count += 1
        try:
            importlib.import_module(module.name)
        except Exception as exc:
            failures.append(f"{module.name}: {exc!r}")
    if failures:
        raise AssertionError("; ".join(failures))
    return f"{count} modules"


def check_environment_steps() -> str:
    """Reset and step a mock-rendered environment.

    Returns:
        Observation keys produced.

    Raises:
        AssertionError: If an observation falls outside the declared space.
    """
    import gs_dronegym

    env = gs_dronegym.make("PointNav-v0", scene=None, renderer_device="cpu")
    obs, _ = env.reset(seed=0)
    if not env.observation_space.contains(obs):
        raise AssertionError("reset observation outside observation_space")
    for _ in range(10):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if not env.observation_space.contains(obs):
            raise AssertionError("step observation outside observation_space")
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return f"observation keys {sorted(obs)}"


def check_seed_once_reproducibility() -> str:
    """Confirm seeding once fixes every later episode.

    Returns:
        Number of episodes compared.

    Raises:
        AssertionError: If two identically seeded runs diverge.
    """
    import gs_dronegym

    def goals() -> np.ndarray:
        env = gs_dronegym.make("PointNav-v0", scene=None, observation_mode="state")
        env.reset(seed=0)
        collected = [np.asarray(env.unwrapped.goal_position, dtype=np.float32).copy()]
        for _ in range(3):
            env.reset()
            collected.append(np.asarray(env.unwrapped.goal_position, dtype=np.float32).copy())
        env.close()
        return np.stack(collected)

    first, second = goals(), goals()
    if not np.array_equal(first, second):
        raise AssertionError("seeded episode sequence is not reproducible")
    return f"{first.shape[0]} episodes identical"


def check_learner_compatible_space() -> str:
    """Confirm the trainable configuration exposes only Box spaces.

    Returns:
        Observation keys in the trainable configuration.

    Raises:
        AssertionError: If a non-Box space remains.
    """
    from gymnasium import spaces

    import gs_dronegym

    env = gs_dronegym.make(
        "PointNav-v0", scene=None, instruction_mode="features", observation_mode="state"
    )
    obs, _ = env.reset(seed=0)
    if not all(isinstance(space, spaces.Box) for space in env.observation_space.spaces.values()):
        raise AssertionError("feature mode still exposes a non-Box space")
    if not env.observation_space.contains(obs):
        raise AssertionError("feature-mode observation outside observation_space")
    env.close()
    return f"all-Box keys {sorted(obs)}"


def check_dataset_factory() -> str:
    """Generate and validate a tiny mock dataset end to end.

    Returns:
        Episode and step counts.

    Raises:
        AssertionError: If the generated dataset fails validation.
    """
    from gs_dronegym import (
        DatasetGenerationConfig,
        SceneSelectionConfig,
        generate_dataset,
        validate_generated_dataset,
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / "dataset"
        generate_dataset(
            DatasetGenerationConfig(
                output_root=root,
                scene_selection=SceneSelectionConfig(sources=("mock://smoke_a", "mock://smoke_b")),
                episodes_per_scene=2,
                image_size=(32, 32),
                renderer_device="cpu",
                seed=0,
                allow_mock_rendering=True,
                dataset_id="wheel_smoke",
            )
        )
        report = validate_generated_dataset(root)
        if not report.valid:
            raise AssertionError(f"generated dataset failed validation: {report}")
        return f"{report.n_episodes} episodes, {report.n_steps} steps"


def check_console_scripts() -> str:
    """Confirm every declared console script is installed and runs.

    Returns:
        Number of scripts executed.

    Raises:
        AssertionError: If a script is missing or exits with an error.
    """
    distribution = importlib.metadata.distribution(DISTRIBUTION)
    installed = {ep.name for ep in distribution.entry_points if ep.group == "console_scripts"}
    if installed != EXPECTED_SCRIPTS:
        raise AssertionError(
            f"missing {sorted(EXPECTED_SCRIPTS - installed)}, "
            f"unexpected {sorted(installed - EXPECTED_SCRIPTS)}"
        )
    scripts_dir = sysconfig.get_path("scripts")
    failures = []
    for name in sorted(installed):
        executable = shutil.which(name, path=scripts_dir)
        if executable is None:
            failures.append(f"{name}: not found in {scripts_dir}")
            continue
        result = subprocess.run(
            [executable, "--help"], capture_output=True, text=True, timeout=600, check=False
        )
        if result.returncode != 0:
            failures.append(f"{name}: exit {result.returncode}: {result.stderr.strip()[-400:]}")
    if failures:
        raise AssertionError("; ".join(failures))
    return f"{len(installed)} scripts ran --help"


def check_version_consistency() -> str:
    """Confirm the runtime version matches the installed distribution metadata.

    The version is declared in both ``pyproject.toml`` and
    ``gs_dronegym/__init__.py``; a release that bumps only one would report the
    wrong version at runtime.

    Returns:
        The agreed version string.

    Raises:
        AssertionError: If ``gs_dronegym.__version__`` disagrees with the metadata.
    """
    import gs_dronegym

    metadata_version = importlib.metadata.version(DISTRIBUTION)
    if gs_dronegym.__version__ != metadata_version:
        raise AssertionError(
            f"gs_dronegym.__version__ is {gs_dronegym.__version__} but the installed "
            f"distribution is {metadata_version}"
        )
    return metadata_version


CHECKS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("version consistency", check_version_consistency),
    ("required modules", check_required_modules),
    ("every module imports", check_every_module_imports),
    ("environment steps", check_environment_steps),
    ("seed-once reproducibility", check_seed_once_reproducibility),
    ("learner-compatible space", check_learner_compatible_space),
    ("dataset factory", check_dataset_factory),
    ("console scripts", check_console_scripts),
)


def main() -> int:
    """Run every smoke check against the installed wheel.

    Returns:
        Process exit code.
    """
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    try:
        print(f"PASS installed location: {check_installed_location()}")
    except Exception:
        print("FAIL installed location; aborting so no check runs against source files")
        traceback.print_exc()
        return 1

    failed = 0
    for name, check in CHECKS:
        try:
            print(f"PASS {name}: {check()}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(CHECKS) + 1 - failed}/{len(CHECKS) + 1} smoke checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
