"""Portable path behavior and actionable missing-data errors."""

from pathlib import Path
import importlib

import pytest

from src.data.preprocessing import load_raw_data
from src.utils.paths import DATA_ROOT_ENV, PROJECT_ROOT, get_data_root, require_raw_dataset


def test_project_root_is_independent_of_current_directory(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert PROJECT_ROOT == Path(__file__).resolve().parent.parent
    assert get_data_root() == PROJECT_ROOT / "dataset"


def test_external_data_root_environment_variable_is_resolved(tmp_path, monkeypatch):
    configured = tmp_path / "original dataset"
    monkeypatch.setenv(DATA_ROOT_ENV, str(configured))
    assert get_data_root() == configured.resolve()


def test_relative_external_data_root_is_relative_to_project_not_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV, "shared/original-data")
    monkeypatch.chdir(tmp_path)
    assert get_data_root() == (PROJECT_ROOT / "shared/original-data").resolve()


def test_missing_raw_data_error_is_actionable_and_does_not_create_directories(tmp_path, monkeypatch):
    missing_root = tmp_path / "not-present"
    monkeypatch.setenv(DATA_ROOT_ENV, str(missing_root))
    with pytest.raises(FileNotFoundError) as error:
        require_raw_dataset()
    message = str(error.value)
    assert str(missing_root / "BC_cardiotox_clinical_variables.csv") in message
    assert DATA_ROOT_ENV in message
    assert not missing_root.exists()


def test_data_loader_uses_configured_root_and_reports_missing_file(tmp_path, monkeypatch):
    configured = tmp_path / "external-data"
    monkeypatch.setenv(DATA_ROOT_ENV, str(configured))
    with pytest.raises(FileNotFoundError, match="original cardiotoxicity dataset"):
        load_raw_data()
    assert not configured.exists()


def test_importing_data_modules_does_not_create_directories(monkeypatch):
    import src.data.dataset_exploration as exploration
    import src.data.image_preprocessing as images
    import src.data.preprocessing as preprocessing
    import src.data.quality_assessment as quality
    import src.data.splitting as splitting

    created = []
    original_mkdir = Path.mkdir

    def record_mkdir(path, *args, **kwargs):
        created.append(Path(path))
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    for module in (exploration, images, preprocessing, quality, splitting):
        importlib.reload(module)
    assert created == []
