import logging
import os
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def is_git_lfs_installed():
    """
    Check if Git LFS is installed.
    """
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Git LFS is installed.")
            return True
        else:
            logger.warning("Git LFS is not installed.")
            return False
    except FileNotFoundError:
        logger.error("Git is not installed.")
        return False


def clone_repository(repo_url: str, target_directory: Path):
    """
    Clone a Git repository to the specified target directory.
    """
    if os.path.exists(target_directory):
        logger.warning(
            f"Target directory '{target_directory}' already exists. Skipping clone."
        )
        return

    try:
        subprocess.run(
            ["git", "clone", repo_url, target_directory],
            check=True,
        )
        logger.info(f"Repository cloned into '{target_directory}'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e}")
        raise
