from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REEXEC_ENV_VAR = "PDF_LAYOUT_LAB_PROJECT_VENV_REEXECED"
DISABLE_ENV_VAR = "PDF_LAYOUT_LAB_DISABLE_PROJECT_VENV"


def project_venv_python(project_root: str | Path) -> Path:
    venv_dir = Path(project_root) / ".venv"
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def should_reexec_project_venv(
    project_root: str | Path,
    current_executable: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    env = os.environ if environ is None else environ
    if env.get(DISABLE_ENV_VAR) or env.get(REEXEC_ENV_VAR):
        return False

    venv_python = project_venv_python(project_root)
    if not venv_python.exists():
        return False

    current = Path(current_executable or sys.executable)
    current_path = os.path.normcase(os.path.abspath(os.fspath(current)))
    venv_path = os.path.normcase(os.path.abspath(os.fspath(venv_python)))
    return current_path != venv_path


def exec_project_venv_if_available(project_root: str | Path, argv: Sequence[str] | None = None) -> None:
    if not should_reexec_project_venv(project_root):
        return

    venv_python = project_venv_python(project_root)
    args = [str(venv_python), *(argv or sys.argv)]
    env = os.environ.copy()
    env[REEXEC_ENV_VAR] = "1"
    os.execve(str(venv_python), args, env)
