import os
import pytest
import views_r2darts2
import logging

logger = logging.getLogger(__name__)

def pytest_sessionstart(session):
    """
    Workspace Integrity Check:
    Ensures that the library being tested is the one in the current workspace.
    This prevents 'Ghost Imports' from stale temp folders in other projects.
    """
    expected_path = os.path.abspath(os.path.join(os.getcwd(), "views_r2darts2"))
    actual_path = os.path.dirname(os.path.abspath(views_r2darts2.__file__))
    
    if actual_path.lower() != expected_path.lower():
        error_msg = (
            "\n\n🚨 WORKSPACE INTEGRITY FAILURE 🚨\n"
            "Environment Contamination Detected!\n"
            f"Expected: {expected_path}\n"
            f"Actual:   {actual_path}\n\n"
            "You are importing 'views_r2darts2' from a location outside this project.\n"
            "Common causes:\n"
            "1. Stale 'temp-views-r2darts2' folders in other model directories.\n"
            "2. PYTHONPATH pointing to a different experiment.\n"
            "3. Package not installed in editable mode (pip install -e .).\n\n"
            "Please run the 'Exorcism Protocol' to clear stale paths."
        )
        pytest.exit(error_msg)
    
    print(f"\n✅ Workspace Integrity Verified: {actual_path}")