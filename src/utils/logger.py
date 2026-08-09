from loguru import logger

from config.settings import LOG_DIR, LOG_LEVEL

LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "resume_analyzer.log",
    rotation="5 MB",
    retention="10 days",
    level=LOG_LEVEL,
    enqueue=True,
)


def get_logger():
    """
    Return the configured application logger.
    """
    return logger