import pytest
import toml
from pathlib import Path
from uncertimety.main import load_config


def create_toml_file(tmp_path: Path, filename: str, data: dict) -> str:
    path = tmp_path / filename
    with open(path, "w") as f:
        toml.dump(data, f)
    return filename


def test_load_valid_config(tmp_path, monkeypatch):
    test_data = {"server": {"host": "localhost", "port": 8080}}
    filename = create_toml_file(tmp_path, "test_config.toml", test_data)

    config = load_config(filename, data_dir=tmp_path)

    assert config["server"]["host"] == "localhost"
    assert config["server"]["port"] == 8080


def test_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config("missing.toml", data_dir=tmp_path)


def test_invalid_toml(tmp_path):
    bad_file = tmp_path / "bad.toml"
    bad_file.write_text("this is not valid = toml:")

    with pytest.raises(toml.TomlDecodeError):
        load_config("bad.toml", data_dir=tmp_path)
