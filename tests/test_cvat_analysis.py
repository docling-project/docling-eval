"""Tests for CVAT annotation analysis tools."""

import json
import os
from pathlib import Path
from typing import List

import pytest
from docling_core.types.doc.document import ContentLayer
from dotenv import load_dotenv

from docling_eval.cvat_tools.analysis import (
    print_containment_tree,
    print_elements_and_paths,
)
from docling_eval.cvat_tools.document import DocumentStructure
from docling_eval.cvat_tools.models import CVATAnnotationPath, CVATElement
from docling_eval.cvat_tools.tree import (
    TreeNode,
    apply_reading_order_to_tree,
    build_global_reading_order,
)
from docling_eval.cvat_tools.validator import Validator

IS_CI = bool(os.getenv("CI"))
load_dotenv()


def create_sample_xml(tmp_path: Path) -> Path:
    """Create a sample CVAT XML file for testing."""
    xml_content = """<?xml version='1.0' encoding='utf-8'?>
<annotations>
<image id="1" name="test.png" width="1600" height="1200">
    <box label="picture" source="manual" occluded="0" xtl="182.39" ytl="176.05" xbr="470.10" ybr="289.41" z_order="0">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="json" />
      <attribute name="type">LOGO</attribute>
    </box>
    <box label="picture" source="manual" occluded="0" xtl="290.15" ytl="374.08" xbr="519.62" ybr="461.15" z_order="1">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="json" />
      <attribute name="type">ARTWORK</attribute>
    </box>
    <box label="section_header" source="manual" occluded="0" xtl="329.04" ytl="400.31" xbr="456.00" ybr="446.80" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="level">2</attribute>
    </box>
    <polyline label="reading_order" source="manual" occluded="0" points="1082.34,894.69;1092.22,976.67;1052.71,1029.01;302.10,695.18;274.44,769.26;278.39,846.30;1324.32,655.68;1361.30,739.80;1323.33,796.91;408.76,427.53;378.90,501.80;426.54,559.88" z_order="3">
      <attribute name="level">2</attribute>
    </polyline>
    <box label="text" source="manual" occluded="0" xtl="294.20" ytl="193.46" xbr="462.10" ybr="275.43" z_order="3">
      <attribute name="content_layer">BODY</attribute>
    </box>
    <box label="picture" source="manual" occluded="0" xtl="190.37" ytl="643.70" xbr="419.84" ybr="730.77" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="json" />
      <attribute name="type">ARTWORK</attribute>
    </box>
    <box label="picture" source="manual" occluded="0" xtl="952.84" ytl="842.22" xbr="1182.30" ybr="929.29" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="json" />
      <attribute name="type">ARTWORK</attribute>
    </box>
    <box label="picture" source="manual" occluded="0" xtl="1184.90" ytl="612.10" xbr="1407.49" ybr="699.17" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="json" />
      <attribute name="type">ARTWORK</attribute>
    </box>
    <box label="text" source="manual" occluded="0" xtl="290.01" ytl="529.23" xbr="747.80" ybr="586.50" z_order="3">
      <attribute name="content_layer">BODY</attribute>
    </box>
    <box label="text" source="manual" occluded="0" xtl="189.79" ytl="799.84" xbr="461.90" ybr="860.10" z_order="3">
      <attribute name="content_layer">BODY</attribute>
    </box>
    <box label="text" source="manual" occluded="0" xtl="735.96" ytl="1007.72" xbr="1196.70" ybr="1041.30" z_order="3">
      <attribute name="content_layer">BODY</attribute>
    </box>
    <box label="text" source="manual" occluded="0" xtl="1028.30" ytl="771.67" xbr="1410.05" ybr="831.94" z_order="3">
      <attribute name="content_layer">BODY</attribute>
    </box>
    <polyline label="reading_order" source="manual" occluded="0" points="346.13,233.79;1121.44,272.47;830.82,471.65" z_order="3">
      <attribute name="level">1</attribute>
    </polyline>
    <box label="picture" source="manual" occluded="0" xtl="184.85" ytl="368.94" xbr="1418.20" ybr="1052.22" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="json" />
      <attribute name="type">INFOGRAPHIC</attribute>
    </box>
    <box label="section_header" source="manual" occluded="0" xtl="190.77" ytl="752.44" xbr="361.20" ybr="793.89" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="level">3</attribute>
    </box>
    <box label="section_header" source="manual" occluded="0" xtl="1023.83" ytl="958.86" xbr="1194.27" ybr="1000.31" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="level">3</attribute>
    </box>
    <box label="section_header" source="manual" occluded="0" xtl="1209.51" ytl="719.85" xbr="1406.60" ybr="761.30" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="level">3</attribute>
    </box>
    <box label="section_header" source="manual" occluded="0" xtl="232.74" ytl="674.86" xbr="359.69" ybr="721.34" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="level">2</attribute>
    </box>
    <box label="section_header" source="manual" occluded="0" xtl="1241.13" ytl="635.35" xbr="1368.08" ybr="681.84" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="level">2</attribute>
    </box>
    <box label="section_header" source="manual" occluded="0" xtl="1019.90" ytl="867.45" xbr="1146.85" ybr="913.94" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="level">2</attribute>
    </box>
    <box label="section_header" source="manual" occluded="0" xtl="290.04" ytl="479.35" xbr="502.90" ybr="520.80" z_order="3">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="level">3</attribute>
    </box>
    <box label="section_header" source="file" occluded="0" xtl="918.60" ytl="167.44" xbr="1446.51" ybr="383.72" z_order="2">
      <attribute name="content_layer">BODY</attribute>
      <attribute name="level">1</attribute>
    </box>
  </image>
</annotations>
    """
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(xml_content)
    return xml_path


def test_document_structure_creation(tmp_path):
    """Test creation and basic properties of DocumentStructure."""
    xml_path = create_sample_xml(tmp_path)
    doc = DocumentStructure.from_cvat_xml(
        xml_path,
        "test.png",
    )

    # Test basic properties
    assert len(doc.elements) == 20  # Total number of boxes
    assert len(doc.paths) == 2  # Two reading order paths
    assert doc.image_info.width == 1600
    assert doc.image_info.height == 1200
    assert doc.image_info.name == "test.png"

    # Test element properties
    element = doc.get_element_by_id(0)
    assert element is not None
    assert element.label == "picture"
    assert element.attributes["type"] == "LOGO"
    assert element.content_layer == ContentLayer.BODY

    # Test path properties
    path = doc.get_path_by_id(0)
    assert path is not None
    assert path.label == "reading_order"
    assert path.level == 2
    assert len(path.points) == 12  # Number of points in the first reading order path


def test_path_mappings(tmp_path):
    """Test path mappings in DocumentStructure."""
    xml_path = create_sample_xml(tmp_path)
    doc = DocumentStructure.from_cvat_xml(
        xml_path,
        "test.png",
    )

    # Test reading order mappings
    assert len(doc.path_mappings.reading_order) == 2  # Two reading order paths
    for path_id, element_ids in doc.path_mappings.reading_order.items():
        assert len(element_ids) > 0
        for element_id in element_ids:
            element = doc.get_element_by_id(element_id)
            assert element is not None
            assert element.content_layer == ContentLayer.BODY

    # Test path-to-container mappings
    # assert len(doc.path_to_container) == 2  # One container per reading order path
    for path_id, container in doc.path_to_container.items():
        assert container is not None
        assert container.element is not None
        assert container.element.content_layer == ContentLayer.BODY


def test_analysis_functions(tmp_path):
    """Test analysis functions with DocumentStructure."""
    xml_path = create_sample_xml(tmp_path)
    doc = DocumentStructure.from_cvat_xml(
        xml_path,
        "test.png",
    )

    # Test printing elements and paths
    print("\n=== Elements and Paths ===")
    print_elements_and_paths(doc.elements, doc.paths, doc.image_info)

    # Test printing containment tree
    print("\n=== Containment Tree ===")
    print_containment_tree(doc.tree_roots, doc.image_info)

    # Test applying reading order and printing the reordered tree
    print("\n=== Containment Tree (Reading Order Applied) ===")
    global_ro = build_global_reading_order(
        doc.paths,
        doc.path_mappings.reading_order,
        doc.path_to_container,
        doc.tree_roots,
    )
    apply_reading_order_to_tree(doc.tree_roots, global_ro)
    print_containment_tree(doc.tree_roots, doc.image_info)


def test_validation_report(tmp_path):
    """Test validation report generation for DocumentStructure."""
    xml_path = create_sample_xml(tmp_path)
    doc = DocumentStructure.from_cvat_xml(
        xml_path,
        "test.png",
    )

    validator = Validator()
    validation_report = validator.validate_sample("test.png", doc)
    print(validation_report.model_dump_json(exclude_none=True, indent=2))
