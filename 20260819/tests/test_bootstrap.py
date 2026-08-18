import tempfile
import unittest
from pathlib import Path

from pdf_layout_lab.bootstrap import (
    DISABLE_ENV_VAR,
    REEXEC_ENV_VAR,
    project_venv_python,
    should_reexec_project_venv,
)


class BootstrapTests(unittest.TestCase):
    def test_reexecs_when_project_venv_exists_and_current_python_differs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_python = project_venv_python(temp_dir)
            venv_python.parent.mkdir(parents=True)
            venv_python.touch()

            self.assertTrue(
                should_reexec_project_venv(
                    temp_dir,
                    current_executable="/usr/bin/python",
                    environ={},
                )
            )

    def test_does_not_reexec_when_already_using_project_venv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_python = project_venv_python(temp_dir)
            venv_python.parent.mkdir(parents=True)
            venv_python.touch()

            self.assertFalse(
                should_reexec_project_venv(
                    temp_dir,
                    current_executable=venv_python,
                    environ={},
                )
            )

    def test_reexecs_even_when_venv_python_points_to_same_binary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            real_python = Path(temp_dir) / "python-real"
            real_python.touch()
            venv_python = project_venv_python(temp_dir)
            venv_python.parent.mkdir(parents=True)
            try:
                venv_python.symlink_to(real_python)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")

            self.assertTrue(
                should_reexec_project_venv(
                    temp_dir,
                    current_executable=real_python,
                    environ={},
                )
            )

    def test_does_not_reexec_without_project_venv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(
                should_reexec_project_venv(
                    temp_dir,
                    current_executable="/usr/bin/python",
                    environ={},
                )
            )

    def test_env_flags_disable_reexec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_python = project_venv_python(temp_dir)
            venv_python.parent.mkdir(parents=True)
            venv_python.touch()

            for env_var in (DISABLE_ENV_VAR, REEXEC_ENV_VAR):
                with self.subTest(env_var=env_var):
                    self.assertFalse(
                        should_reexec_project_venv(
                            temp_dir,
                            current_executable="/usr/bin/python",
                            environ={env_var: "1"},
                        )
                    )


if __name__ == "__main__":
    unittest.main()
