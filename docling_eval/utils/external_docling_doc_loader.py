from pathlib import Path
from typing import Optional

from docling_core.types.doc.document import DoclingDocument


class ExternalDoclingDocLoader:
    def __init__(self, external_predictions_dir: Path):
        self._external_predictions_dir = external_predictions_dir

    def __call__(self, doc_id: str) -> Optional[DoclingDocument]:
        r"""
        Load the DoclingDocument from the external predictions path
        """
        json_path = self._external_predictions_dir / f"{doc_id}.json"
        dt_path = self._external_predictions_dir / f"{doc_id}.dt"
        yaml_path = self._external_predictions_dir / f"{doc_id}.yaml"
        yml_path = self._external_predictions_dir / f"{doc_id}.yml"

        if json_path.is_file():
            return DoclingDocument.load_from_json(json_path)
        if dt_path.is_file():
            return DoclingDocument.load_from_doctags(dt_path)
        if yaml_path.is_file():
            return DoclingDocument.load_from_yaml(yaml_path)
        if yml_path.is_file():
            return DoclingDocument.load_from_yaml(yml_path)
        return None
