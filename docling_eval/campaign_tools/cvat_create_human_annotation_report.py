#!/usr/bin/env python3

"""
CLI utility to scan a CVAT project for frames annotated by humans (detected by a special label) and export a CSV report.

This tool connects to a CVAT instance, scans all tasks in a given project for frames annotated with a specific label
(used as a human annotation indicator), and outputs a CSV file listing all such frames. This is typically used after
annotation or review steps to audit or analyze human annotation coverage.

Usage:
    uv run python scratches/scratch_51.py \
        --server-url <cvat_url> \
        --username <username> \
        --password <password> \
        --project-id <project_id> \
        --human-label <label_name> \
        [--output-csv <output.csv>] \
        [--debug]

Arguments:
    server_url:   CVAT server URL (e.g., https://cvat.example.com/)
    username:     CVAT username
    password:     CVAT password
    project_id:   ID of the CVAT project to scan
    human_label:  Name of the label that only humans would annotate (case-insensitive), e.g. "reading_order"
    output_csv:   Output CSV filename (default: human_annotated_frames.csv)
    debug:        Enable debug logging

CSV Output Columns:
    frame_filename:  Filename of the annotated frame
    task_name:       Name of the CVAT task
    dataset_name:    Dataset name (prefix of task name)
    frame_url:       Direct URL to the frame in CVAT
"""

import csv
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

import typer
from cvat_sdk import make_client
from cvat_sdk.exceptions import CvatSdkException

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


class CVATHumanAnnotationDetector:
    """Detect frames that have been touched by human annotators in CVAT."""

    def __init__(
        self,
        server_url: str,
        username: str,
        password: str,
        project_id: int,
        human_indicator_label: str,
    ):
        """Initialize the CVAT Human Annotation Detector.

        Args:
            server_url: CVAT server URL
            username: CVAT username
            password: CVAT password
            project_id: ID of the project to analyze
            human_indicator_label: Label that indicates human annotation
        """
        self.server_url = server_url.rstrip("/")
        self.project_id = project_id
        self.human_indicator_label = (
            human_indicator_label.lower()
        )  # Case-insensitive matching

        # Initialize CVAT client
        logger.info(f"Connecting to CVAT server at {server_url}")
        self.client = make_client(host=server_url, credentials=(username, password))

        # Results storage
        self.human_annotated_frames: list[dict[str, object]] = (
            []
        )  # List of dicts with frame info
        self.tasks_info: dict[int, dict[str, object]] = {}  # task_id -> task_info

        # Statistics
        self.stats: dict[str, Any] = {
            "total_tasks": 0,
            "total_jobs": 0,
            "total_frames": 0,
            "human_annotated_frames": 0,
            "tasks_with_human_annotations": 0,
            "jobs_with_human_annotations": 0,
            "found_labels": set(),
        }

    def get_project_tasks(self):
        """Get all tasks for the specified project."""
        logger.info(f"Fetching tasks for project {self.project_id}...")

        # Use the high-level API to filter tasks by project_id
        all_tasks = self.client.tasks.list()
        project_tasks = [
            task for task in all_tasks if task.project_id == self.project_id
        ]
        logger.info(f"Found {len(project_tasks)} tasks in project {self.project_id}")

        self.stats["total_tasks"] = len(project_tasks)

        # Store basic task info
        for task in project_tasks:
            # Extract dataset name from task name (part before _task_xyz)
            dataset_name = (
                task.name.split("_task_")[0] if "_task_" in task.name else task.name
            )

            self.tasks_info[task.id] = {
                "id": task.id,
                "name": task.name,
                "dataset_name": dataset_name,
                "status": task.status,
                "size": task.size,
                "human_annotated_frames_count": 0,
            }

        return project_tasks

    def get_task_frame_names(self, task) -> Dict[int, str]:
        """Get frame ID to filename mapping for a task.

        Args:
            task: CVAT task object

        Returns:
            dict: Mapping from frame_id to frame_filename
        """
        frame_names = {}
        try:
            # Get frame metadata containing actual filenames
            frames_info = task.get_frames_info()
            logger.debug(f"Retrieved {len(frames_info)} frames info for task {task.id}")

            # Create mapping from frame ID to actual filename
            for frame_id, frame_meta in enumerate(frames_info):
                frame_names[frame_id] = frame_meta.name

            logger.debug(
                f"Created frame name mapping for task {task.id}: {len(frame_names)} frames"
            )

        except Exception as e:
            logger.warning(f"Failed to get frame names for task {task.id}: {str(e)}")
            logger.warning("Falling back to generic frame naming")
            # Fallback to generic naming if frames info is not available
            for i in range(task.size):
                frame_names[i] = f"frame_{i:06d}"

        return frame_names

    def detect_human_annotations(self):
        """Find all frames with human annotations across all tasks in the project."""
        tasks = self.get_project_tasks()

        for task in tasks:
            logger.info(f"Processing task {task.id}: {task.name}")
            self.process_task(task)

        # Update task-level statistics
        tasks_with_annotations = set()
        for frame_info in self.human_annotated_frames:
            tasks_with_annotations.add(frame_info["task_id"])

        self.stats["tasks_with_human_annotations"] = len(tasks_with_annotations)
        self.stats["human_annotated_frames"] = len(self.human_annotated_frames)

        return self.human_annotated_frames

    def process_task(self, task):
        """Process a single task to find human-annotated frames."""
        try:
            # Get jobs for this task
            jobs = list(task.get_jobs())
            self.stats["total_jobs"] += len(jobs)

            # Get actual frame names from task metadata
            frame_names = self.get_task_frame_names(task)

            # Process each job
            for job in jobs:
                logger.info(f"Processing job {job.id} for task {task.id}")
                self.process_job(task, job, frame_names)

        except CvatSdkException as e:
            logger.error(f"CVAT SDK error processing task {task.id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {str(e)}")
            raise

    def process_job(self, task, job, frame_names):
        """Process a single job to find human-annotated frames."""
        try:
            # Retrieve job annotations - returns LabeledData object
            annotations = self.client.api_client.jobs_api.retrieve_annotations(job.id)[
                0
            ]
            logger.debug(f"Retrieved annotations of type: {type(annotations).__name__}")

            # Find frames with annotations having the target label
            human_frames: Set[int] = self.find_human_annotated_frames(annotations)

            # Count total frames in job
            job_size = job.stop_frame - job.start_frame + 1
            self.stats["total_frames"] += job_size

            # Process human annotated frames
            if human_frames:
                self.stats["jobs_with_human_annotations"] += 1
                logger.info(
                    f"Found {len(human_frames)} human-annotated frames in job {job.id}"
                )

                # Add frame info to results
                for frame_id in human_frames:
                    # Convert job-relative frame ID to task-relative frame ID
                    global_frame_id = job.start_frame + frame_id

                    # Get actual frame filename from mapping
                    frame_filename = frame_names.get(
                        global_frame_id, f"frame_{global_frame_id:06d}"
                    )

                    # Create frame info record
                    frame_info = {
                        "task_id": task.id,
                        "job_id": job.id,
                        "frame_id": global_frame_id,
                        "frame_filename": frame_filename,
                        "task_name": task.name,
                        "dataset_name": self.tasks_info[task.id]["dataset_name"],
                        "url": f"{self.server_url}/tasks/{task.id}/jobs/{job.id}?frame={global_frame_id}",
                    }

                    self.human_annotated_frames.append(frame_info)
            else:
                logger.info(f"No human-annotated frames found in job {job.id}")

        except CvatSdkException as e:
            logger.error(f"CVAT SDK error processing job {job.id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing job {job.id}: {str(e)}")
            raise

    def find_human_annotated_frames(self, annotations) -> Set[int]:
        """Find frames with human annotations.

        Args:
            annotations: Annotations object from the CVAT SDK

        Returns:
            set: Set of frame IDs with human annotations
        """
        human_frames: Set[int] = set()
        target_label_id = self.get_target_label_id()

        if target_label_id is None:
            logger.debug(
                f"Human indicator label '{self.human_indicator_label}' not found in project labels"
            )
            return human_frames

        # Process shapes (including polylines)
        if "shapes" in annotations:
            for shape in annotations["shapes"]:
                # Check if this shape has our target label
                if shape["label_id"] == target_label_id:
                    frame_id = shape["frame"]
                    shape_type = shape.get("type", "")
                    human_frames.add(frame_id)
                    logger.debug(
                        f"Found human annotation with shape type '{shape_type}' on frame {frame_id}"
                    )

        # Process tracks (for tracked objects)
        if "tracks" in annotations:
            for track in annotations["tracks"]:
                # Check if this track has our target label
                if track["label_id"] == target_label_id:
                    # Process track shapes
                    if "shapes" in track and track["shapes"]:
                        for track_shape in track["shapes"]:
                            frame_id = track_shape["frame"]
                            human_frames.add(frame_id)
                            logger.debug(
                                f"Found human annotation in track on frame {frame_id}"
                            )

        return human_frames

    def get_target_label_id(self) -> Optional[int]:
        """Get the ID of the human indicator label from the project.

        Returns:
            int or None: The ID of the human indicator label if found, None otherwise
        """
        all_labels = []
        page = 1
        page_size = 100  # Adjust as needed

        # Use the LabelsApi to fetch all labels for the project, handling pagination
        try:
            while True:
                # Fetch the current page of labels
                labels_data, _ = self.client.api_client.labels_api.list(
                    project_id=self.project_id, page=page, page_size=page_size
                )

                # Add the current page's results to our collection
                all_labels.extend(labels_data.results)

                # Check if there are more pages
                if not labels_data.next:
                    break

                # Move to the next page
                page += 1

            # Log the total number of labels found
            logger.debug(
                f"Found {len(all_labels)} labels for project {self.project_id}"
            )

            # Search for our target label
            for label in all_labels:
                label_name = label.name.lower()
                self.stats["found_labels"].add(label_name)

                if label_name == self.human_indicator_label:
                    logger.debug(
                        f"Found target label '{label_name}' with ID {label.id}"
                    )
                    return label.id

            # If we get here, we didn't find the label
            if all_labels:
                logger.debug(
                    f"Available labels: {[label.name for label in all_labels]}"
                )

        except Exception as e:
            logger.error(
                f"Error fetching labels for project {self.project_id}: {str(e)}"
            )

        return None

    def save_csv_results(self, filename="human_annotated_frames.csv"):
        """Save results in CSV format as specified.

        CSV format: frame_filename,task_name,dataset_name,frame_url
        """
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(
                ["frame_filename", "task_name", "dataset_name", "frame_url"]
            )

            # Write data rows
            for frame in self.human_annotated_frames:
                writer.writerow(
                    [
                        frame["frame_filename"],
                        frame["task_name"],
                        frame["dataset_name"],
                        frame["url"],
                    ]
                )

        logger.info(
            f"CSV results saved to {filename} with {len(self.human_annotated_frames)} rows"
        )

    def generate_summary(self):
        """Generate a summary of the detection process."""
        summary = []
        summary.append(
            f"Human Annotations Detection Summary for Project {self.project_id}"
        )
        summary.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Human indicator label: {self.human_indicator_label}")
        summary.append("")
        summary.append(f"Total tasks: {self.stats['total_tasks']}")
        summary.append(f"Total jobs: {self.stats['total_jobs']}")
        summary.append(f"Total frames: {self.stats['total_frames']}")
        summary.append(
            f"Tasks with human annotations: {self.stats['tasks_with_human_annotations']}"
        )
        summary.append(
            f"Jobs with human annotations: {self.stats['jobs_with_human_annotations']}"
        )
        summary.append(
            f"Total human-annotated frames: {self.stats['human_annotated_frames']}"
        )

        # Add found labels in debug mode
        if logging.getLogger().level == logging.DEBUG:
            summary.append("")
            summary.append("Labels found in the data:")
            for label in sorted(self.stats["found_labels"]):
                summary.append(f"- {label}")

        return "\n".join(summary)


@app.command()
def detect_human_annotations_cli(
    server_url: str = typer.Option(
        ..., help="CVAT server URL (e.g., https://cvat.example.com/)"
    ),
    username: str = typer.Option(..., help="CVAT username"),
    password: str = typer.Option(..., help="CVAT password"),
    project_id: int = typer.Option(..., help="ID of the CVAT project to scan."),
    human_label: str = typer.Option(
        ..., help="Label name that indicates human annotation (case-insensitive)."
    ),
    output_csv: str = typer.Option(
        "human_annotated_frames.csv", help="Output CSV filename."
    ),
    debug: bool = typer.Option(False, help="Enable debug logging."),
) -> None:
    """
    Scan a CVAT project for frames annotated by humans (detected by a special label) and export a CSV report.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    detector = CVATHumanAnnotationDetector(
        server_url=server_url,
        username=username,
        password=password,
        project_id=project_id,
        human_indicator_label=human_label,
    )
    detector.detect_human_annotations()
    detector.save_csv_results(output_csv)
    print(detector.generate_summary())
    logger.info("Human annotation detection complete.")
    logger.info(
        f"Found {detector.stats['human_annotated_frames']} human-annotated frames across {detector.stats['tasks_with_human_annotations']} tasks."
    )


if __name__ == "__main__":
    app()
