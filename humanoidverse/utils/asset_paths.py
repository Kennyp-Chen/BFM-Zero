from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any


PACKAGE_SCHEME = "package://"


def _as_str(path: Any) -> str:
    path_text = "" if path is None else str(path)
    if path_text.startswith("package:/") and not path_text.startswith(PACKAGE_SCHEME):
        return path_text.replace("package:/", PACKAGE_SCHEME, 1)
    return path_text


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_package_roots(package_name: str) -> list[Path]:
    candidates: list[Path] = []
    env_names = [f"{package_name.upper()}_ROOT"]
    if package_name == "ht_urdf":
        env_names.append("HT_URDF_ROOT")

    for env_name in dict.fromkeys(env_names):
        env_root = os.environ.get(env_name)
        if env_root:
            root = Path(env_root).expanduser()
            candidates.extend([root, root / package_name, root / package_name / package_name])

    workspace_parent = _repo_root().parent
    home = Path.home()
    candidates.extend(
        [
            workspace_parent / package_name / package_name,
            workspace_parent / package_name,
            home / package_name / package_name,
            home / package_name,
        ]
    )

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)
    return unique_candidates


def _fallback_package_root(package_name: str) -> Path | None:
    existing_roots = [candidate for candidate in _candidate_package_roots(package_name) if candidate.exists()]
    for candidate in existing_roots:
        if (candidate / "__init__.py").exists():
            return candidate.resolve()
    if existing_roots:
        return existing_roots[0].resolve()
    return None


def _join_package_path(package_root: Path, package_name: str, rel_path: str) -> Path:
    if not rel_path:
        return package_root

    rel = Path(rel_path)
    path = package_root / rel
    if rel.parts[:1] == (package_name,) and package_root.name == package_name:
        normalized_path = package_root.joinpath(*rel.parts[1:])
        if normalized_path.exists() or not path.exists():
            return normalized_path
    return path


def _package_path(path: str) -> Path | None:
    if not path.startswith(PACKAGE_SCHEME):
        return None

    package_and_rel = path[len(PACKAGE_SCHEME) :]
    package_name, _, rel_path = package_and_rel.partition("/")
    if not package_name:
        raise ValueError(f"Invalid package asset path: {path!r}")

    try:
        module = importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if exc.name != package_name:
            raise
        package_root = _fallback_package_root(package_name)
        if package_root is None:
            raise
    else:
        if module.__file__ is None:
            package_root = _fallback_package_root(package_name)
            if package_root is None:
                raise ValueError(f"Package {package_name!r} does not expose a filesystem path.")
        else:
            package_root = Path(module.__file__).resolve().parent

    return _join_package_path(package_root, package_name, rel_path)


def resolve_asset_path(asset_root: Any, asset_file: Any = "") -> Path:
    """Resolve local or package-scoped robot asset paths.

    Package paths use the form ``package://ht_urdf/path/in/package``. They
    resolve from the installed package when available, then fall back to
    ``HT_URDF_ROOT`` and common sibling checkout locations.
    """
    root = _as_str(asset_root)
    file = _as_str(asset_file)

    file_package_path = _package_path(file)
    if file_package_path is not None:
        return file_package_path

    if file and os.path.isabs(file):
        return Path(file)

    root_package_path = _package_path(root)
    if root_package_path is not None:
        return root_package_path / file

    return Path(os.path.join(root, file))
