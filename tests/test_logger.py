import logging
import pytest
from uncertimety.logger import init_logger


@pytest.fixture
def test_logger(tmp_path):
    # Initialize logger writing to tmp_path, but file existence won't be tested
    return init_logger(name="test_logger", log_dir=tmp_path, log_file="test.log")


def test_logger_has_handlers(test_logger):
    handlers = test_logger.handlers
    assert any(
        isinstance(h, logging.StreamHandler) for h in handlers
    ), "Missing console handler"
    assert any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers
    ), "Missing rotating file handler"


def test_logger_level(test_logger):
    assert test_logger.level == logging.DEBUG


def test_logger_emits_log_message(test_logger, caplog):
    caplog.set_level(logging.INFO, logger=test_logger.name)
    test_msg = "Logger test message"
    test_logger.info(test_msg)
    # Check that the message is in captured logs
    assert any(test_msg in record.message for record in caplog.records)


def test_logger_no_duplicate_handlers(test_logger):
    handler_ids = [id(h) for h in test_logger.handlers]
    assert len(handler_ids) == len(set(handler_ids)), "Duplicate handlers found"
