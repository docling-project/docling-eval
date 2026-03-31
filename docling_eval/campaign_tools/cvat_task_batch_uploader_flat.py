#!/usr/bin/env python3

"""Upload flat folder layouts to CVAT — each subfolder becomes one task.

Expected input layout:
    Annotations/
      <task_name>/
        images/
          *.jpg|*.jpeg|*.png|*.tif|*.tiff|*.bmp|*.webp
        annotations.xml

Each first-level folder is uploaded as one CVAT task.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from cvat_sdk.core.client import Client, Config
from cvat_sdk.core.proxies.tasks import ResourceType, Task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ANNOTATION_FILENAME = "annotations.xml"
IMAGES_DIRNAME = "images"
IGNORED_NAMES = {".DS_Store"}


class StrictFlatFolderTaskUploader:
    """Uploader for strict `folder/images + folder/annotations.xml` layout."""

    def __init__(
        self,
        server_url: str,
        username: str,
        password: str,
        project_id: int,
        org_slug: Optional[str],
        update_existing: bool,
        verify_ssl: bool,
    ) -> None:
        self.project_id = project_id
        self.update_existing = update_existing

        self.client = Client(url=server_url, config=Config(verify_ssl=verify_ssl))
        self.client.login((username, password))

        if org_slug:
            self.client.organization_slug = org_slug

        self._existing_tasks: Optional[dict[str, Task]] = None

    def get_existing_tasks(self) -> dict[str, Task]:
        if self._existing_tasks is None:
            logger.info("Fetching existing tasks for project %s", self.project_id)
            project_tasks = [
                task
                for task in self.client.tasks.list()
                if task.project_id == self.project_id
            ]
            self._existing_tasks = {task.name: task for task in project_tasks}
            logger.info("Found %d existing tasks", len(self._existing_tasks))

        return self._existing_tasks

    def upload_directory(self, input_directory: Path) -> None:
        if not input_directory.is_dir():
            raise ValueError(
                f"Input directory does not exist or is not a directory: {input_directory}"
            )

        document_dirs = sorted(
            path
            for path in input_directory.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

        if not document_dirs:
            logger.warning("No document folders found in %s", input_directory)
            return

        logger.info("Found %d document folders", len(document_dirs))
        for document_dir in document_dirs:
            self._upload_document_folder(document_dir)

    def _upload_document_folder(self, document_dir: Path) -> None:
        self._validate_document_layout(document_dir)

        task_name = document_dir.name
        existing_tasks = self.get_existing_tasks()
        annotation_path = document_dir / ANNOTATION_FILENAME

        if task_name in existing_tasks:
            if not self.update_existing:
                logger.info("Task '%s' exists. Skipping.", task_name)
                return

            logger.info("Task '%s' exists. Importing annotations only.", task_name)
            existing_tasks[task_name].import_annotations(
                format_name="CVAT 1.1",
                filename=str(annotation_path),
            )
            return

        image_files = self._collect_image_files(document_dir / IMAGES_DIRNAME)
        logger.info("Creating task '%s' with %d images", task_name, len(image_files))

        task = self.client.tasks.create_from_data(
            spec={"name": task_name, "project_id": self.project_id},
            resource_type=ResourceType.LOCAL,
            resources=[str(path) for path in image_files],
            data_params={"image_quality": 100, "sorting_method": "lexicographical"},
        )
        logger.info("Created task '%s' (id=%s)", task_name, task.id)

        task.import_annotations(format_name="CVAT 1.1", filename=str(annotation_path))

        if self._existing_tasks is not None:
            self._existing_tasks[task_name] = task

    def _validate_document_layout(self, document_dir: Path) -> None:
        images_dir = document_dir / IMAGES_DIRNAME
        annotation_file = document_dir / ANNOTATION_FILENAME

        if not images_dir.is_dir():
            raise ValueError(f"Missing required directory: {images_dir}")

        if not annotation_file.is_file():
            raise ValueError(f"Missing required annotation file: {annotation_file}")

        allowed = {IMAGES_DIRNAME, ANNOTATION_FILENAME, *IGNORED_NAMES}
        extras = sorted(p.name for p in document_dir.iterdir() if p.name not in allowed)
        if extras:
            raise ValueError(
                f"Unexpected files/folders in {document_dir}. "
                f"Allowed: {sorted(allowed)}. Found extra: {extras}"
            )

        if not self._collect_image_files(images_dir):
            raise ValueError(f"No supported images found in {images_dir}")

    def _collect_image_files(self, images_dir: Path) -> list[Path]:
        image_files = [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        image_files.sort()
        return image_files


@app.command()
def upload_flat_document_tasks(
    input_directory: Path = typer.Option(
        ...,
        help="Directory whose immediate subfolders each represent one CVAT task.",
    ),
    server_url: str = typer.Option(
        ..., help="CVAT server URL, e.g. https://cvat.example.com/"
    ),
    username: str = typer.Option(..., help="CVAT username"),
    password: str = typer.Option(..., help="CVAT password"),
    project_id: int = typer.Option(..., help="Target CVAT project id"),
    org_slug: Optional[str] = typer.Option(None, help="CVAT organization slug"),
    update_existing: bool = typer.Option(
        False,
        help="If enabled, existing tasks are not recreated and only annotations.xml is re-imported.",
    ),
    verify_ssl: bool = typer.Option(
        False,
        "--verify-ssl/--no-verify-ssl",
        help="Enable TLS certificate verification.",
    ),
) -> None:
    """Upload strict flat document-folder tasks to CVAT."""
    uploader = StrictFlatFolderTaskUploader(
        server_url=server_url,
        username=username,
        password=password,
        project_id=project_id,
        org_slug=org_slug,
        update_existing=update_existing,
        verify_ssl=verify_ssl,
    )
    uploader.upload_directory(input_directory)
    logger.info("Done uploading document folders")


if __name__ == "__main__":
    app()
