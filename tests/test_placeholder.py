def test_project_structure():
    from pathlib import Path

    assert Path("src/__init__.py").exists()
    assert Path("api").exists()
