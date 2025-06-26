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


def create_sample_xml_with_doublepage_list(tmp_path: Path) -> Path:
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
      <annotations>
        <image id="4" name="doc_6b18af59b633f89b96a64aa435e0f7616eb1813d884c4c3da5e4cea9a8f9316b_ps_000001_pe_000003.png" width="2448" height="1584">
          <box label="section_header" source="file" occluded="0" xtl="106.00" ytl="206.37" xbr="639.85" ybr="236.67" z_order="0">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1330.00" ytl="182.65" xbr="1706.07" ybr="230.07" z_order="0">
            <attribute name="level">1</attribute>
          </box>
          <box label="text" source="file" occluded="0" xtl="106.00" ytl="319.70" xbr="583.98" ybr="502.12" z_order="1">
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1366.01" ytl="252.65" xbr="1813.70" ybr="435.06" z_order="1">
            <attribute name="level">1</attribute>
          </box>
          <box label="text" source="file" occluded="0" xtl="106.00" ytl="524.69" xbr="579.59" ybr="923.10" z_order="2">
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1330.00" ytl="457.64" xbr="1728.68" ybr="478.06" z_order="2">
            <attribute name="level">1</attribute>
          </box>
          <box label="picture" source="file" occluded="0" xtl="0.01" ytl="974.90" xbr="594.86" ybr="1405.76" z_order="3">
            <attribute name="json"></attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1330.00" ytl="492.63" xbr="1814.60" ybr="567.06" z_order="3">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1366.01" ytl="589.63" xbr="1732.70" ybr="637.05" z_order="4">
            <attribute name="level">1</attribute>
          </box>
          <box label="text" source="file" occluded="0" xtl="630.00" ytl="319.68" xbr="1071.90" ybr="367.24" z_order="4">
          </box>
          <box label="list_item" source="file" occluded="0" xtl="630.00" ytl="389.70" xbr="1103.96" ybr="437.12" z_order="5">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1366.01" ytl="651.63" xbr="1571.85" ybr="672.05" z_order="5">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="630.00" ytl="451.69" xbr="1083.66" ybr="580.11" z_order="6">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1366.01" ytl="686.62" xbr="1620.22" ybr="707.05" z_order="6">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="630.00" ytl="594.69" xbr="1086.23" ybr="615.11" z_order="7">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1366.01" ytl="721.62" xbr="1496.57" ybr="742.05" z_order="7">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="630.00" ytl="629.69" xbr="1063.18" ybr="704.11" z_order="8">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1366.01" ytl="756.62" xbr="1500.81" ybr="777.04" z_order="8">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="666.01" ytl="726.68" xbr="1090.05" ybr="774.10" z_order="9">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="1366.01" ytl="791.62" xbr="1609.60" ybr="812.04" z_order="9">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="666.01" ytl="788.68" xbr="1123.26" ybr="890.10" z_order="10">
            <attribute name="level">1</attribute>
          </box>
          <box label="section_header" source="file" occluded="0" xtl="1854.00" ytl="184.47" xbr="2201.40" ybr="210.24" z_order="10">
            <attribute name="level">1</attribute>
          </box>
          <box label="list_item" source="file" occluded="0" xtl="666.01" ytl="904.67" xbr="1101.35" ybr="1060.09" z_order="11">
            <attribute name="level">1</attribute>
          </box>
          <box label="text" source="file" occluded="0" xtl="1854.00" ytl="222.90" xbr="2336.51" ybr="513.31" z_order="11">
          </box>
          <box label="list_item" source="file" occluded="0" xtl="666.01" ytl="1074.67" xbr="1114.46" ybr="1203.08" z_order="12">
            <attribute name="level">1</attribute>
          </box>
          <box label="text" source="file" occluded="0" xtl="1854.00" ytl="535.89" xbr="2346.80" ybr="799.30" z_order="12">
          </box>
          <box label="page_footer" source="file" occluded="0" xtl="106.00" ytl="1499.81" xbr="306.35" ybr="1514.96" z_order="13">
          </box>
          <box label="picture" source="file" occluded="0" xtl="1328.20" ytl="893.66" xbr="2447.04" ybr="1403.71" z_order="13">
            <attribute name="json"></attribute>
          </box>
          <box label="page_footer" source="file" occluded="0" xtl="306.35" ytl="1499.82" xbr="559.64" ybr="1514.87" z_order="14">
          </box>
          <box label="page_footer" source="file" occluded="0" xtl="1330.00" ytl="1499.81" xbr="1529.07" ybr="1514.96" z_order="14">
          </box>
          <box label="page_footer" source="file" occluded="0" xtl="1003.54" ytl="1499.82" xbr="1118.00" ybr="1514.87" z_order="15">
          </box>
          <box label="page_footer" source="file" occluded="0" xtl="1529.07" ytl="1499.82" xbr="1782.35" ybr="1514.87" z_order="15">
          </box>
          <polyline label="group" source="manual" occluded="0" points="923.25,405.34;876.99,508.96;938.05,607.03;889.95,666.24;1369.18,205.51;1406.19,468.25;1365.48,529.31" z_order="16">
          </polyline>
          <box label="page_footer" source="file" occluded="0" xtl="2227.54" ytl="1499.82" xbr="2342.00" ybr="1514.87" z_order="16">
          </box>
          <polyline label="group" source="manual" occluded="0" points="963.96,749.50;1021.32,840.17;951.01,956.74;1030.57,1132.53" z_order="16">
          </polyline>
          <polyline label="group" source="manual" occluded="0" points="1419.14,616.28;1435.80,664.39;1415.44,694.00;1419.14,736.55;1400.64,768.01;1419.14,805.02" z_order="16">
          </polyline>
          <polyline label="reading_order" source="manual" occluded="0" points="332.99,218.46;246.03,462.70;353.35,768.01;210.87,1188.04;836.29,331.33;878.84,407.19;814.08,507.11;865.89,605.18;815.93,677.34;858.49,745.80;821.48,825.37;895.50,978.95;838.14,1130.68;481.02,1245.40;270.08,1508.14;412.56,1506.29;1060.18,1506.29;1539.42,199.95;1606.03,346.13;1524.61,468.25;1546.82,538.57;1500.56,616.28;1519.06,660.69;1491.31,694.00;1435.80,731.00;1465.40,766.16;1437.65,799.46;1926.14,192.55;2029.76,372.04;1918.74,655.14;2039.01,1156.58;1432.10,1506.29;1696.69,1510.00;2275.85,1508.14" z_order="16">
          </polyline>
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
    doc.tree_roots = apply_reading_order_to_tree(doc.tree_roots, global_ro)
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


def test_doublepage_list(tmp_path):
    """Test doublepage list in DocumentStructure."""
    xml_path = create_sample_xml_with_doublepage_list(tmp_path)
    doc = DocumentStructure.from_cvat_xml(
        xml_path,
        "doc_6b18af59b633f89b96a64aa435e0f7616eb1813d884c4c3da5e4cea9a8f9316b_ps_000001_pe_000003.png",
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
    doc.tree_roots = apply_reading_order_to_tree(doc.tree_roots, global_ro)
    print_containment_tree(doc.tree_roots, doc.image_info)

    validator = Validator()
    validation_report = validator.validate_sample(
        "doc_6b18af59b633f89b96a64aa435e0f7616eb1813d884c4c3da5e4cea9a8f9316b_ps_000001_pe_000003.png",
        doc,
    )
    print(validation_report.model_dump_json(exclude_none=True, indent=2))
