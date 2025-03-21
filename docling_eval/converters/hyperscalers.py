import json
from pathlib import Path

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.document import (
    DoclingDocument,
    PageItem,
    ProvenanceItem,
    TableCell,
    TableData,
    TableItem,
)
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image


def convert_textract_output_to_docling(response, image_path: Path) -> DoclingDocument:
    """Converts Textract output to DoclingDocument."""
    doc_id = Path(image_path).stem
    document = DoclingDocument(name=doc_id)

    # document.origin = DocumentOrigin(
    #     mimetype="image/tiff",
    #     binary_hash=hash(json.dumps(response)),
    #     filename=image_path.name,
    # )

    page_no = str(response.get("DocumentMetadata", {}).get("Pages", 1))

    for block in response.get("Blocks", []):
        if block["BlockType"] == "PAGE":
            width = block["Geometry"]["BoundingBox"]["Width"]
            height = block["Geometry"]["BoundingBox"]["Height"]
            document.pages[int(page_no)] = PageItem(
                size=Size(width=float(width), height=float(height)),
                page_no=int(page_no),
            )

    for block in response.get("Blocks", []):
        if block["BlockType"] == "LINE":
            bbox = BoundingBox(
                l=block["Geometry"]["BoundingBox"]["Left"],
                t=block["Geometry"]["BoundingBox"]["Top"],
                r=block["Geometry"]["BoundingBox"]["Left"]
                + block["Geometry"]["BoundingBox"]["Width"],
                b=block["Geometry"]["BoundingBox"]["Top"]
                + block["Geometry"]["BoundingBox"]["Height"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            text = block.get("Text", "")
            prov = ProvenanceItem(
                page_no=int(page_no), bbox=bbox, charspan=(0, len(text))
            )

            document.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif block["BlockType"] == "TABLE":
            bbox = BoundingBox(
                l=block["Geometry"]["BoundingBox"]["Left"],
                t=block["Geometry"]["BoundingBox"]["Top"],
                r=block["Geometry"]["BoundingBox"]["Left"]
                + block["Geometry"]["BoundingBox"]["Width"],
                b=block["Geometry"]["BoundingBox"]["Top"]
                + block["Geometry"]["BoundingBox"]["Height"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            prov = ProvenanceItem(page_no=int(page_no), bbox=bbox, charspan=(0, 0))

            table_data = TableData(table_cells=[], num_rows=0, num_cols=0, grid=[])

            cell_blocks = [
                b
                for b in response.get("Blocks", [])
                if b["BlockType"] == "CELL" and b.get("ParentId", "") == block["Id"]
            ]

            for cell_block in cell_blocks:
                cell_text = ""
                line_blocks_in_cell = [
                    b
                    for b in response.get("Blocks", [])
                    if b["BlockType"] == "LINE"
                    and b.get("ParentId", "") == cell_block["Id"]
                ]

                for line_block in line_blocks_in_cell:
                    cell_text += line_block.get("Text", "") + " "

                cell = TableCell(
                    bbox=BoundingBox(
                        l=cell_block["Geometry"]["BoundingBox"]["Left"],
                        t=cell_block["Geometry"]["BoundingBox"]["Top"],
                        r=cell_block["Geometry"]["BoundingBox"]["Left"]
                        + cell_block["Geometry"]["BoundingBox"]["Width"],
                        b=cell_block["Geometry"]["BoundingBox"]["Top"]
                        + cell_block["Geometry"]["BoundingBox"]["Height"],
                        coord_origin=CoordOrigin.TOPLEFT,
                    ),
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=0,
                    end_row_offset_idx=0,
                    start_col_offset_idx=0,
                    end_col_offset_idx=0,
                    text=cell_text.strip(),
                    column_header=False,
                    row_header=False,
                    row_section=False,
                )

                table_data.table_cells.append(cell)

            document.add_table(label=DocItemLabel.TABLE, prov=prov, data=table_data)

    return document


def convert_google_output_to_docling(document, image_path):
    """Converts Google Document AI output to DoclingDocument format."""
    doc_id = Path(image_path).stem
    doc = DoclingDocument(name=doc_id)

    # doc.origin = DocumentOrigin(
    #     mimetype="image/tiff",
    #     binary_hash=hash(json.dumps(document)),
    #     filename=Path(image_path).name,
    # )

    for page in document.get("pages", []):
        page_no = page.get("pageNumber", 1)

        # Create page with dimensions
        page_width = page.get("dimension", {}).get("width", 0)
        page_height = page.get("dimension", {}).get("height", 0)
        doc.pages[page_no] = PageItem(
            size=Size(width=float(page_width), height=float(page_height)),
            page_no=page_no,
        )

        for paragraph in page.get("paragraphs", []):
            # Extract text content from text_anchor and text_segments
            text_content = ""
            if "layout" in paragraph and "textAnchor" in paragraph["layout"]:
                for text_segment in paragraph["layout"]["textAnchor"].get(
                    "textSegments", []
                ):
                    if "endIndex" in text_segment:
                        start_index = int(text_segment.get("startIndex", 0))
                        end_index = int(text_segment.get("endIndex", 0))
                        if document.get("text") and start_index < len(document["text"]):
                            text_content += document["text"][start_index:end_index]

            # Extract paragraph bounding box
            para_bbox = extract_bbox_from_vertices(
                paragraph.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
            )

            bbox_obj = BoundingBox(
                l=para_bbox["l"],
                t=para_bbox["t"],
                r=para_bbox["r"],
                b=para_bbox["b"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            prov = ProvenanceItem(
                page_no=page_no, bbox=bbox_obj, charspan=(0, len(text_content))
            )

            doc.add_text(label=DocItemLabel.PARAGRAPH, text=text_content, prov=prov)

        for table in page.get("tables", []):
            table_bbox = extract_bbox_from_vertices(
                table.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
            )

            num_rows = len(table.get("headerRows", [])) + len(table.get("bodyRows", []))
            num_cols = 0  # Will be calculated based on cells

            # Create table bounding box
            table_bbox_obj = BoundingBox(
                l=table_bbox["l"],
                t=table_bbox["t"],
                r=table_bbox["r"],
                b=table_bbox["b"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            table_prov = ProvenanceItem(
                page_no=page_no, bbox=table_bbox_obj, charspan=[]
            )

            table_data = TableData(
                table_cells=[],
                num_rows=num_rows,
                num_cols=0,  # Will update as we process cells
                grid=[],
            )

            for row_index, row in enumerate(table.get("headerRows", [])):
                num_cols = max(table_data.num_cols, len(row.get("cells", [])))
                table_data.num_cols = num_cols

                process_table_row(row, row_index, document, table_data, is_header=True)

            header_row_count = len(table.get("headerRows", []))
            for row_index, row in enumerate(table.get("bodyRows", [])):
                actual_row_index = header_row_count + row_index
                num_cols = max(table_data.num_cols, len(row.get("cells", [])))
                table_data.num_cols = num_cols

                process_table_row(
                    row, actual_row_index, document, table_data, is_header=False
                )

            doc.add_table(table_data=table_data, prov=table_prov)

    return doc


def extract_bbox_from_vertices(vertices):
    """Helper function to extract bbox coordinates from vertices."""
    if len(vertices) >= 4:
        return {
            "l": vertices[0].get("x", 0),
            "t": vertices[0].get("y", 0),
            "r": vertices[2].get("x", 0),
            "b": vertices[2].get("y", 0),
        }
    return {"l": 0, "t": 0, "r": 0, "b": 0}


def process_table_row(row, row_index, document, table_data, is_header=False):
    """Process a table row and add cells to table_data."""
    for cell_index, cell in enumerate(row.get("cells", [])):
        cell_text_content = ""
        if "layout" in cell and "textAnchor" in cell["layout"]:
            for text_segment in cell["layout"]["textAnchor"].get("textSegments", []):
                start_index = int(text_segment.get("startIndex", 0))
                end_index = int(text_segment.get("endIndex", 0))
                if document.get("text") and start_index < len(document["text"]):
                    cell_text_content += document["text"][start_index:end_index]
        cell_bbox = extract_bbox_from_vertices(
            cell.get("layout", {}).get("boundingPoly", {}).get("vertices", [])
        )
        row_span = cell.get("rowSpan", 1)
        col_span = cell.get("colSpan", 1)

        table_cell = TableCell(
            bbox=BoundingBox(
                l=cell_bbox["l"],
                t=cell_bbox["t"],
                r=cell_bbox["r"],
                b=cell_bbox["b"],
                coord_origin=CoordOrigin.TOPLEFT,
            ),
            row_span=row_span,
            col_span=col_span,
            start_row_offset_idx=row_index,
            end_row_offset_idx=row_index + row_span - 1,
            start_col_offset_idx=cell_index,
            end_col_offset_idx=cell_index + col_span - 1,
            text=cell_text_content.strip(),
            column_header=is_header,
            row_header=not is_header
            and cell_index == 0,  # First column might be row header
            row_section=False,
        )

        table_data.table_cells.append(table_cell)


def convert_azure_output_to_docling(analyze_result, image_path) -> DoclingDocument:
    """Converts Azure Document Intelligence output to DoclingDocument format."""
    doc_id = Path(image_path).parent.stem
    doc = DoclingDocument(name=doc_id)
    try:
        w, h = Image.open(image_path).size
    except Exception:
        # Default if image can't be opened
        w, h = 0, 0

    for page in analyze_result.get("pages", []):
        page_no = page.get("page_number", 1)

        page_width = page.get("width", w)
        page_height = page.get("height", h)
        doc.pages[page_no] = PageItem(
            size=Size(width=float(page_width), height=float(page_height)),
            page_no=page_no,
        )

        for word in page.get("words", []):
            polygon = word.get("polygon", [])
            bbox = extract_bbox_from_polygon(polygon)

            text_content = word.get("content", "")

            bbox_obj = BoundingBox(
                l=bbox["l"],
                t=bbox["t"],
                r=bbox["r"],
                b=bbox["b"],
                coord_origin=CoordOrigin.TOPLEFT,
            )

            prov = ProvenanceItem(
                page_no=page_no, bbox=bbox_obj, charspan=(0, len(text_content))
            )

            doc.add_text(label=DocItemLabel.TEXT, text=text_content, prov=prov)

    for table in analyze_result.get("tables", []):
        page_no = table.get("page_range", {}).get("first_page_number", 1)
        row_count = table.get("row_count", 0)
        col_count = table.get("column_count", 0)

        table_bounds = (
            table.get("boundingRegions", [{}])[0]
            if table.get("boundingRegions")
            else {}
        )
        table_polygon = table_bounds.get("polygon", [])
        table_bbox = extract_bbox_from_polygon(table_polygon)

        table_bbox_obj = BoundingBox(
            l=table_bbox["l"],
            t=table_bbox["t"],
            r=table_bbox["r"],
            b=table_bbox["b"],
            coord_origin=CoordOrigin.TOPLEFT,
        )

        table_prov = ProvenanceItem(page_no=page_no, bbox=table_bbox_obj, charspan=[])

        table_data = TableData(
            table_cells=[], num_rows=row_count, num_cols=col_count, grid=[]
        )

        for cell in table.get("cells", []):

            cell_text = cell.get("content", "").strip()
            row_index = cell.get("row_index", 0)
            col_index = cell.get("column_index", 0)
            row_span = cell.get("row_span", 1)
            col_span = cell.get("column_span", 1)

            cell_bounds = (
                cell.get("boundingRegions", [{}])[0]
                if cell.get("boundingRegions")
                else {}
            )
            cell_polygon = cell_bounds.get("polygon", [])
            cell_bbox = extract_bbox_from_polygon(cell_polygon)

            table_cell = TableCell(
                bbox=BoundingBox(
                    l=cell_bbox["l"],
                    t=cell_bbox["t"],
                    r=cell_bbox["r"],
                    b=cell_bbox["b"],
                    coord_origin=CoordOrigin.TOPLEFT,
                ),
                row_span=row_span,
                col_span=col_span,
                start_row_offset_idx=row_index,
                end_row_offset_idx=row_index + row_span - 1,
                start_col_offset_idx=col_index,
                end_col_offset_idx=col_index + col_span - 1,
                text=cell_text,
                column_header=False,
                row_header=False,
                row_section=False,
            )

            table_data.table_cells.append(table_cell)

        doc.add_table(label=DocItemLabel.TABLE, prov=table_prov, data=table_data)

    return doc


def extract_bbox_from_polygon(polygon):
    """Helper function to extract bbox coordinates from polygon data."""
    if not polygon or not isinstance(polygon, list):
        return {"l": 0, "t": 0, "r": 0, "b": 0}

    # Handle flat array format: [x1, y1, x2, y2, x3, y3, x4, y4]
    if len(polygon) >= 8 and all(isinstance(p, (int, float)) for p in polygon):
        return {"l": polygon[0], "t": polygon[1], "r": polygon[4], "b": polygon[5]}
    # Handle array of point objects: [{x, y}, {x, y}, ...]
    elif len(polygon) >= 4 and all(
        isinstance(p, dict) and "x" in p and "y" in p for p in polygon
    ):
        return {
            "l": polygon[0]["x"],
            "t": polygon[0]["y"],
            "r": polygon[2]["x"],
            "b": polygon[2]["y"],
        }
    else:
        return {"l": 0, "t": 0, "r": 0, "b": 0}


def extract_schema(d, depth=10):
    if isinstance(d, dict):
        schema = {}
        for key, value in d.items():
            schema[key] = extract_schema(value, depth + 1)
        return schema
    elif isinstance(d, list) and d:
        return [extract_schema(d[0], depth + 1)]  # Take only one example from the list
    else:
        return "value"  # Placeholder for actual values
