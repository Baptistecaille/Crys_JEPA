import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scripts_can_import_project_package_from_repo_root():
    for script, args in [
        ("train_mvp.py", ["--help"]),
        ("evaluate_mvp.py", ["--help"]),
        ("infer_cif.py", ["--help"]),
    ]:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
