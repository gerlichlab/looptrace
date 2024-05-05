"""Test that the document for pipeline execution control is up-to-date."""

from pathlib import Path
import subprocess

__author__ = "Vince Reuter"
__credits__ = ["Vince Reuter"]


PROJECT_ROOT = Path(__file__).parent.parent


def test_pipline_control_flow_doc_is_current(tmp_path):
    # Pretests
    target = tmp_path / "doc.md"
    assert not target.exists()
    current = PROJECT_ROOT / "docs" / "pipeline-execution-control-and-rerun.md"
    assert current.is_file()
    
    # Create snapshot of target doc.
    builder_program = PROJECT_ROOT / "bin" / "cli" / "generate_excution_control_document.py"
    cmd = ["python", str(builder_program), "-O", str(target)]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    # Check that the current doc matches that snapshot (current code state).
    assert target.is_file()
    with open(target, "r") as fh:
        snapshot_lines = fh.readlines()
    with open(current, "r") as fh:
        current_lines = fh.readlines()
    keep_line = lambda line: not line.startswith("<!--")
    snapshot_lines = list(filter(keep_line, snapshot_lines))
    current_lines = list(filter(keep_line, current_lines))
    assert snapshot_lines == current_lines
