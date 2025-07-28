#!/usr/bin/env python3

"""
CLI utility to upload prepared CVAT task directories to a CVAT instance for annotation and inspection.

This tool takes a set of prepared CVAT tasks directories, organized in the structure produced by the
`create_annotation_tasks_from_folders.py` tool, and uploads them to a specified CVAT instance.

Usage:
    uv run python scratches/scratch_50.py \
        --input-directory <input_dir> \
        --server-url <cvat_url> \
        --username <username> \
        --password <password> \
        --project-id <project_id> \
        [--org-slug <org_slug>] \
        [--update-existing] \
        [--force-update-annotations]

Arguments:
    input_directory: Root directory containing dataset subdirectories (each with cvat_dataset_preannotated/cvat_tasks)
    server_url:      CVAT server URL (e.g., https://cvat.example.com/)
    username:        CVAT username
    password:        CVAT password
    project_id:      ID of the existing CVAT project to upload tasks to
    org_slug:        (Optional) Organization slug for CVAT API calls
    update_existing: (Flag) Update tasks that already exist (default: False)
    force_update_annotations: (Flag) Update annotations on existing tasks (default: False)

Example:
    uv run python scratches/scratch_50.py \
        --input-directory /path/to/datasets/ \
        --server-url https://cvat.example.com/ \
        --username user \
        --password pass \
        --project-id 42 \
        --org-slug myorg \
        --update-existing \
        --force-update-annotations
"""

import glob
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from cvat_sdk import make_client
from cvat_sdk.core.client import Client, Config
from cvat_sdk.core.proxies.tasks import ResourceType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


class CVATTaskUploader:
    """Class to handle uploading tasks and annotations to CVAT."""

    def __init__(
        self,
        server_url: str,
        username: str,
        password: str,
        project_id: int,
        org_slug: Optional[str],
        update_existing: bool = False,
        force_update_annotations: bool = False,
    ) -> None:
        """Initialize the CVAT Task Uploader.

        Args:
            server_url: CVAT server URL
            username: CVAT username
            password: CVAT password
            project_id: ID of the existing project to upload tasks to
            org_slug: Organization slug for CVAT API calls
            update_existing: Whether to update tasks that already exist
            force_update_annotations: Whether to update annotations on existing tasks
        """
        self.project_id = project_id
        self.org_slug = org_slug
        self.update_existing = update_existing
        self.force_update_annotations = force_update_annotations

        # Initialize CVAT client with insecure TLS
        config = Config(verify_ssl=False)
        self.client = Client(url=server_url, config=config)
        self.client.login((username, password))

        # Set organization context only if provided
        if self.org_slug is not None:
            self.client.organization_slug = self.org_slug

        # Cache of existing tasks to avoid repeated API calls
        self._existing_tasks = None

    def get_existing_tasks(self):
        """Get all existing tasks for the project.

        Returns:
            dict: Dictionary mapping task names to task objects
        """
        if self._existing_tasks is None:
            logger.info(f"Fetching existing tasks for project {self.project_id}...")
            # Get all tasks and filter for those belonging to our project
            all_tasks = self.client.tasks.list()
            project_tasks = [
                task for task in all_tasks if task.project_id == self.project_id
            ]
            self._existing_tasks = {task.name: task for task in project_tasks}
            logger.info(f"Found {len(self._existing_tasks)} existing tasks")

        return self._existing_tasks

    def process_datasets(self, base_dir: str) -> None:
        """Process all dataset directories.

        Args:
            base_dir: Base directory containing dataset subdirectories
        """
        logger.info(f"Processing datasets in {base_dir}")

        # Get all subdirectories in the base directory
        subset_dirs = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        for subset in subset_dirs:
            self.process_subset(base_dir, subset)

    def process_subset(self, base_dir: str, subset: str) -> None:
        """Process a specific dataset subset.

        Args:
            base_dir: Base directory containing dataset subdirectories
            subset: Name of the subset to process
        """
        logger.info(f"Processing subset: {subset}")

        tasks_dir = os.path.join(
            base_dir, subset, "cvat_dataset_preannotated", "cvat_tasks"
        )

        if not os.path.isdir(tasks_dir):
            logger.warning(f"No cvat_tasks directory found in {tasks_dir}")
            return

        # Get all task directories (using glob to match task_* pattern)
        task_dirs = sorted(glob.glob(os.path.join(tasks_dir, "task_*")))

        for task_dir in task_dirs:
            if os.path.isdir(task_dir):
                task_name = os.path.basename(task_dir)
                full_task_name = f"{subset}_{task_name}"

                self.create_and_upload_task(
                    tasks_dir, task_dir, task_name, full_task_name
                )

    def create_and_upload_task(
        self, tasks_dir: str, task_dir: str, task_name: str, full_task_name: str
    ) -> None:
        """Create a task and upload images and annotations if it doesn't exist.

        Args:
            tasks_dir: Directory containing all tasks for this subset
            task_dir: Directory for this specific task
            task_name: Name of the task (e.g., "task_00")
            full_task_name: Full name of the task (e.g., "page_vibrant_task_00")
        """
        # Check if task already exists
        existing_tasks = self.get_existing_tasks()
        if full_task_name in existing_tasks:
            if not self.update_existing:
                logger.info(f"Task '{full_task_name}' already exists. Skipping.")

                # Check if we should update annotations for existing task
                if self.force_update_annotations:
                    self.upload_annotations(
                        existing_tasks[full_task_name], tasks_dir, task_name
                    )
                return
            else:
                logger.info(f"Task '{full_task_name}' already exists. Updating.")
                # In the current version of CVAT SDK, we can't update task images,
                # so we'll just update annotations
                self.upload_annotations(
                    existing_tasks[full_task_name], tasks_dir, task_name
                )
                return

        logger.info(f"Creating new task: {full_task_name}")

        # Find all images in the task directory (supporting multiple image formats)
        image_files = []
        for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
            image_files.extend(glob.glob(os.path.join(task_dir, f"*{ext}")))

        # Sort images lexicographically
        image_files.sort()

        if not image_files:
            logger.warning(f"No images found in {task_dir}")
            return

        # Create the task using project labels
        try:
            task_spec = {"name": full_task_name, "project_id": self.project_id}

            # Parameters for data upload
            data_params = {
                "image_quality": 100,  # Use maximum quality
                "sorting_method": "lexicographical",  # Ensure lexicographical sorting
            }

            # Create the task with images
            task = self.client.tasks.create_from_data(
                spec=task_spec,
                resource_type=ResourceType.LOCAL,
                resources=image_files,
                data_params=data_params,
            )

            logger.info(f"Created task with ID: {task.id}")

            # Upload annotations for the new task
            self.upload_annotations(task, tasks_dir, task_name)

            # Update the task cache
            if self._existing_tasks is not None:
                self._existing_tasks[full_task_name] = task

        except Exception as e:
            logger.error(f"Error creating task {full_task_name}: {str(e)}")

    def upload_annotations(self, task, tasks_dir: str, task_name: str) -> None:
        """Upload annotations for a task.

        Args:
            task: Task object
            tasks_dir: Directory containing annotation files
            task_name: Base name of the task
        """
        # Find and upload the preannotation file
        preannotation_file = os.path.join(tasks_dir, f"{task_name}_preannotate.xml")

        if os.path.isfile(preannotation_file):
            logger.info(f"Uploading preannotations from {preannotation_file}")
            try:
                task.import_annotations(
                    format_name="CVAT 1.1", filename=preannotation_file
                )
                logger.info("Preannotations uploaded successfully")
            except Exception as e:
                logger.error(f"Error uploading annotations: {str(e)}")
        else:
            logger.warning(f"No preannotation file found at {preannotation_file}")


@app.command()
def upload_cvat_tasks(
    input_directory: Path = typer.Option(
        ..., help="Root directory containing dataset subdirectories."
    ),
    server_url: str = typer.Option(
        ..., help="CVAT server URL (e.g., https://cvat.example.com/)"
    ),
    username: str = typer.Option(..., help="CVAT username"),
    password: str = typer.Option(..., help="CVAT password"),
    project_id: int = typer.Option(
        ..., help="ID of the existing CVAT project to upload tasks to."
    ),
    org_slug: Optional[str] = typer.Option(
        None, help="Organization slug for CVAT API calls (optional)."
    ),
    update_existing: bool = typer.Option(
        False, help="Update tasks that already exist (default: False)."
    ),
    force_update_annotations: bool = typer.Option(
        False, help="Update annotations on existing tasks (default: False)."
    ),
) -> None:
    """
    Upload prepared CVAT task directories to a CVAT instance for annotation and inspection.
    """
    uploader = CVATTaskUploader(
        server_url=server_url,
        username=username,
        password=password,
        project_id=project_id,
        org_slug=org_slug,
        update_existing=update_existing,
        force_update_annotations=force_update_annotations,
    )
    uploader.process_datasets(str(input_directory))
    logger.info("All tasks have been uploaded to CVAT")


if __name__ == "__main__":
    app()
