from pathlib import Path

import pytest
from pydantic import ValidationError

from ramem.config import AppConfig, load_config


def test_default_config_is_valid() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert isinstance(config, AppConfig)
    assert config.context.token_budget == 2048
    assert config.context.total_token_budget == 4096
    assert config.generation.provider == "auto"


def test_unknown_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"surprise": True})
