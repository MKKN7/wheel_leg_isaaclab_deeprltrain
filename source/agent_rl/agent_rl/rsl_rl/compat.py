"""Compatibility helpers for multiple rsl-rl releases.

Recent Isaac Lab/RSL-RL releases removed ``rsl_rl.utils.store_code_state``
although the custom DreamWaQ/HIM/NP3O runners still call it.  Keep the
training dependency within the Isaac Lab-supported range and provide the
small missing helper locally instead of monkey-patching the installed package.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _repo_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(result.stdout.strip()) if result.stdout.strip() else None


def store_code_state(log_dir: str | Path | None, repo_files) -> list[str]:
    """Save git status/diff snapshots and return paths suitable for loggers.

    The function mirrors the old rsl-rl helper's observable contract.  Missing
    repositories or non-git installs are skipped rather than aborting training.
    """
    if not log_dir:
        return []
    output_dir = Path(log_dir).expanduser() / "git"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    seen: set[Path] = set()
    for value in repo_files or []:
        source = Path(str(value)).expanduser()
        root = _repo_root(source)
        if root is None or root in seen:
            continue
        seen.add(root)
        stem = root.name or "repository"
        status_path = output_dir / f"{stem}_status.txt"
        diff_path = output_dir / f"{stem}_diff.patch"
        try:
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
            diff = subprocess.run(
                ["git", "-C", str(root), "diff", "--binary"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
            status_path.write_text(status, encoding="utf-8")
            diff_path.write_text(diff, encoding="utf-8")
        except (OSError, subprocess.CalledProcessError):
            continue
        saved.extend([str(status_path), str(diff_path)])
    return saved


try:
    from rsl_rl.utils import store_code_state as _upstream_store_code_state
except ImportError:  # rsl-rl 5.x
    _upstream_store_code_state = None


if _upstream_store_code_state is not None:
    store_code_state = _upstream_store_code_state


__all__ = ["store_code_state"]
