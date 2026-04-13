import pytest
from PIL import Image
from docling_core.types.doc.document import DoclingDocument

from docling_eval.datamodels.dataset_record import DatasetRecordWithBBox


def test_dataset_record_with_bbox_roundtrip():
    record = DatasetRecordWithBBox(
        document_id="doc-1",
        GroundTruthDocument=DoclingDocument(name="test"),
        GroundTruthPageImages=[Image.new("RGB", (100, 100), "white")],
        GroundTruthBboxOnPageImages={
            0: [
                {
                    "label": "table",
                    "category_id": 3,
                    "bbox": [10, 20, 30, 40],
                    "ltrb": [10, 20, 40, 60],
                }
            ]
        },
    )

    row = record.as_record_dict()

    assert "GroundTruthBboxOnPageImages" in row

    restored = DatasetRecordWithBBox.model_validate(row)
    assert restored.ground_truth_bbox_on_page_images == {
        0: [
            {
                "label": "table",
                "category_id": 3,
                "bbox": [10, 20, 30, 40],
                "ltrb": [10, 20, 40, 60],
            }
        ]
    }


def test_dataset_record_with_bbox_pairs_page_images_with_bboxes():
    first_image = Image.new("RGB", (10, 10), "white")
    second_image = Image.new("RGB", (20, 20), "black")
    record = DatasetRecordWithBBox(
        document_id="doc-1",
        GroundTruthDocument=DoclingDocument(name="test"),
        GroundTruthPageImages=[first_image, second_image],
        GroundTruthBboxOnPageImages={
            1: [
                {
                    "label": "figure",
                    "category_id": 4,
                    "bbox": [1, 2, 3, 4],
                    "ltrb": [1, 2, 4, 6],
                }
            ]
        },
    )

    page_images_with_bboxes = record.get_page_images_with_bboxes()

    assert page_images_with_bboxes == [
        (first_image, []),
        (
            second_image,
            [
                {
                    "label": "figure",
                    "category_id": 4,
                    "bbox": [1, 2, 3, 4],
                    "ltrb": [1, 2, 4, 6],
                }
            ],
        ),
    ]


def test_dataset_record_with_bbox_rejects_non_topleft_origin():
    with pytest.raises(ValueError, match="TOPLEFT"):
        DatasetRecordWithBBox(
            document_id="doc-1",
            GroundTruthDocument=DoclingDocument(name="test"),
            GroundTruthPageImages=[Image.new("RGB", (10, 10), "white")],
            GroundTruthBboxOnPageImages={
                0: [
                    {
                        "coord_origin": "BOTTOMLEFT",
                        "bbox": [1, 1, 2, 2],
                        "ltrb": [1, 1, 3, 3],
                    }
                ]
            },
        )


def test_dataset_record_with_bbox_rejects_box_outside_page_image():
    with pytest.raises(ValueError, match="exceeds page image bounds"):
        DatasetRecordWithBBox(
            document_id="doc-1",
            GroundTruthDocument=DoclingDocument(name="test"),
            GroundTruthPageImages=[Image.new("RGB", (10, 10), "white")],
            GroundTruthBboxOnPageImages={
                0: [
                    {
                        "bbox": [8, 8, 5, 5],
                        "ltrb": [8, 8, 13, 13],
                    }
                ]
            },
        )
