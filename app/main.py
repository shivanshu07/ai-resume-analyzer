from src.utils.logger import get_logger
from src.utils.helper import ensure_directories_exist
from config.settings import (
    RESUME_DIR,
    JD_DIR,
    OUTPUT_DIR,
    LOG_DIR,
)

logger = get_logger()


def initialize_project():
    """
    Initialize required project directories.
    """
    ensure_directories_exist(
        [
            RESUME_DIR,
            JD_DIR,
            OUTPUT_DIR,
            LOG_DIR,
        ]
    )

    logger.info("Project initialized successfully.")


if __name__ == "__main__":
    initialize_project()
    print("Resume Analyzer setup completed successfully.")