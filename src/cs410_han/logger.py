__all__ = ["logger"]

from loguru import logger

from rich.logging import RichHandler

from cs410_han.console import console
from cs410_han.config import settings

log_level = settings.log_level

logger.configure(
    handlers=[
        {
            "sink": RichHandler(
                console=console,
            ),
            "format": "{message}",
            "level": log_level,
        }
    ]
)
