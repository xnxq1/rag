import logging
import sys
from typing import Any

import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO", json_logs: bool = True) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    log_handler = logging.StreamHandler(sys.stdout)

    if json_logs:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            rename_fields={
                "asctime": "timestamp",
                "name": "logger",
                "levelname": "level",
            },
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    log_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(log_handler)
    root_logger.setLevel(log_level)

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
