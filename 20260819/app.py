from pathlib import Path

from pdf_layout_lab.bootstrap import exec_project_venv_if_available


if __name__ == "__main__":
    exec_project_venv_if_available(Path(__file__).resolve().parent)
    from pdf_layout_lab.server import main

    main()
