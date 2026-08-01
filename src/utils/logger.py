from loguru import logger
from config.settings import LOG_DIR

# Ensure logs directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "resume_analyzer.log",
    rotation="5 MB",
    retention="10 days",
    level="INFO",
    enqueue=True,
)

def get_logger():
    return logger