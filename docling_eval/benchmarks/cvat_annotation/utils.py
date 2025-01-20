"""Models for the labels types."""
import os

from enum import Enum
from typing import Tuple
from pathlib import Path

from pydantic import BaseModel

class DocLinkLabel(str, Enum):
    """DocLinkLabel."""

    READING_ORDER = "reading_order"

    TO_CAPTION = "to_caption"
    TO_FOOTNOTE = "to_footnote"
    TO_VALUE = "to_value"

    MERGE = "merge"
    GROUP = "group"

    def __str__(self):
        """Get string value."""
        return str(self.value)

    @staticmethod
    def get_color(label: "DocLinkLabel") -> Tuple[int, int, int]:
        """Return the RGB color associated with a given label."""
        color_map = {
            DocLinkLabel.READING_ORDER: (255, 0, 0),
            DocLinkLabel.TO_CAPTION: (0, 255, 0),
            DocLinkLabel.TO_FOOTNOTE: (0, 255, 0),
            DocLinkLabel.TO_VALUE: (0, 255, 0),
            DocLinkLabel.MERGE: (255, 0, 255),
            DocLinkLabel.GROUP: (255, 255, 0),
        }
        return color_map[label]


class TableComponentLabel(str, Enum):
    """TableComponentLabel."""

    TABLE_ROW = "table_row"  # the most atomic row
    TABLE_COL = "table_column"  # the most atomic col
    TABLE_GROUP = (
        "table_group"  # table-cell group with at least 1 row- or col-span above 1
    )

    def __str__(self):
        """Get string value."""
        return str(self.value)

    @staticmethod
    def get_color(label: "TableComponentLabel") -> Tuple[int, int, int]:
        """Return the RGB color associated with a given label."""
        color_map = {
            TableComponentLabel.TABLE_ROW: (255, 0, 0),
            TableComponentLabel.TABLE_COL: (0, 255, 0),
            TableComponentLabel.TABLE_GROUP: (0, 0, 255),
        }
        return color_map[label]


def rgb_to_hex(r, g, b):
    """
    Converts RGB values to a HEX color code.
    
    Args:
        r (int): Red value (0-255)
        g (int): Green value (0-255)
        b (int): Blue value (0-255)

    Returns:
        str: HEX color code (e.g., "#RRGGBB")
    """
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        raise ValueError("RGB values must be in the range 0-255")
    
    return f"#{r:02X}{g:02X}{b:02X}"


class BenchMarkDirs(BaseModel):

    source_dir: Path
    target_dir: Path
    
    imgs_dir: Path
    bins_dir: Path
    
    page_imgs_dir: Path

    json_true_dir: Path
    json_pred_dir: Path
    json_anno_dir: Path

    html_anno_dir: Path
    html_viz_dir: Path

    project_desc_file: Path
    overview_file: Path

def set_up_directory_structure(source_dir: Path, target_dir: Path) -> BenchMarkDirs:

    source_dir = source_dir
    target_dir = target_dir
    
    imgs_dir = target_dir / "cvat_imgs"
    bins_dir = target_dir / "cvat_bins"

    page_imgs_dir = target_dir / "page_imgs"
    
    json_true_dir = target_dir / "json_groundtruth"
    json_pred_dir = target_dir / "json_predictions"
    json_anno_dir = target_dir / "json_annotations"

    html_anno_dir = target_dir / "html_annotations"
    html_viz_dir = target_dir / "html_annotatations-viz"

    project_desc_file = target_dir / "cvat_description.json"
    overview_file = target_dir / "cvat_overview.json"

    for _ in [
        target_dir,

        imgs_dir,
        bins_dir,

        page_imgs_dir,

        json_true_dir,
        json_pred_dir,
        json_anno_dir,

        html_anno_dir,
        html_viz_dir,
    ]:
        os.makedirs(_, exist_ok=True)    

    result = BenchMarkDirs(
        source_dir=source_dir,
        target_dir=target_dir,

        imgs_dir = imgs_dir,
        bins_dir = bins_dir,

        page_imgs_dir = page_imgs_dir,

        json_true_dir = json_true_dir,
        json_pred_dir = json_pred_dir,
        json_anno_dir = json_anno_dir,

        html_anno_dir = html_anno_dir,
        html_viz_dir = html_viz_dir,

        project_desc_file = project_desc_file,
        overview_file = overview_file,        
    )
    return result
    
