"""Models for the labels types."""

from enum import Enum
from typing import Tuple


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
