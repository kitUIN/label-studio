import io
import os
from enum import Enum

from label_studio_sdk.converter.converter import Format, Converter, logger
from label_studio_sdk.converter.utils import ensure_dir, prettify_result

AiCenterFormat = Enum('AiCenterFormat', list(Format.__members__) + ['PULC'])


def from_string(cls, s):
    try:
        return cls[s]
    except KeyError:
        raise ValueError(f"Invalid value: {s}")


setattr(AiCenterFormat, 'from_string', classmethod(from_string))


class AiCenterConverter(Converter):
    _FORMAT_INFO = {
        Format.JSON: {
            "title": "JSON",
            "description": "List of items in raw JSON format stored in one JSON file. Use to export both the data "
                           "and the annotations for a dataset. It's Label Studio Common Format",
            "link": "https://labelstud.io/guide/export.html#JSON",
        },
        Format.JSON_MIN: {
            "title": "JSON-MIN",
            "description": 'List of items where only "from_name", "to_name" values from the raw JSON format are '
                           "exported. Use to export only the annotations for a dataset.",
            "link": "https://labelstud.io/guide/export.html#JSON-MIN",
        },
        Format.CSV: {
            "title": "CSV",
            "description": "Results are stored as comma-separated values with the column names specified by the "
                           'values of the "from_name" and "to_name" fields.',
            "link": "https://labelstud.io/guide/export.html#CSV",
        },
        Format.TSV: {
            "title": "TSV",
            "description": "Results are stored in tab-separated tabular file with column names specified by "
                           '"from_name" "to_name" values',
            "link": "https://labelstud.io/guide/export.html#TSV",
        },
        Format.CONLL2003: {
            "title": "CONLL2003",
            "description": "Popular format used for the CoNLL-2003 named entity recognition challenge.",
            "link": "https://labelstud.io/guide/export.html#CONLL2003",
            "tags": ["sequence labeling", "text tagging", "named entity recognition"],
        },
        Format.COCO: {
            "title": "COCO",
            "description": "Popular machine learning format used by the COCO dataset for object detection and image "
                           "segmentation tasks with polygons and rectangles.",
            "link": "https://labelstud.io/guide/export.html#COCO",
            "tags": ["image segmentation", "object detection"],
        },
        Format.VOC: {
            "title": "Pascal VOC XML",
            "description": "Popular XML format used for object detection and polygon image segmentation tasks.",
            "link": "https://labelstud.io/guide/export.html#Pascal-VOC-XML",
            "tags": ["image segmentation", "object detection"],
        },
        Format.YOLO: {
            "title": "YOLO",
            "description": "Popular TXT format is created for each image file. Each txt file contains annotations for "
                           "the corresponding image file, that is object class, object coordinates, height & width.",
            "link": "https://labelstud.io/guide/export.html#YOLO",
            "tags": ["image segmentation", "object detection"],
        },
        Format.YOLO_OBB: {
            "title": "YOLOv8 OBB",
            "description": "Popular TXT format is created for each image file. Each txt file contains annotations for "
                           "the corresponding image file. The YOLO OBB format designates bounding boxes by their four corner points "
                           "with coordinates normalized between 0 and 1, so it is possible to export rotated objects.",
            "link": "https://labelstud.io/guide/export.html#YOLO",
            "tags": ["image segmentation", "object detection"],
        },
        Format.BRUSH_TO_NUMPY: {
            "title": "Brush labels to NumPy",
            "description": "Export your brush labels as NumPy 2d arrays. Each label outputs as one image.",
            "link": "https://labelstud.io/guide/export.html#Brush-labels-to-NumPy-amp-PNG",
            "tags": ["image segmentation"],
        },
        Format.BRUSH_TO_PNG: {
            "title": "Brush labels to PNG",
            "description": "Export your brush labels as PNG images. Each label outputs as one image.",
            "link": "https://labelstud.io/guide/export.html#Brush-labels-to-NumPy-amp-PNG",
            "tags": ["image segmentation"],
        },
        Format.ASR_MANIFEST: {
            "title": "ASR Manifest",
            "description": "Export audio transcription labels for automatic speech recognition as the JSON manifest "
                           "format expected by NVIDIA NeMo models.",
            "link": "https://labelstud.io/guide/export.html#ASR-MANIFEST",
            "tags": ["speech recognition"],
        },
        AiCenterFormat.PULC: {
            "title": "Paddle PULC",
            "description": "The results are stored as \\t-separated values with column names specified by the values of the “from_name” and “to_name” fields.",
            "link": "https://github.com/PaddlePaddle/PaddleClas/blob/release/2.6/docs/zh_CN/training/PULC.md",
            "tags": ["image segmentation"],
        },
    }

    def __init__(
            self,
            config,
            project_dir,
            output_tags=None,
            upload_dir=None,
            download_resources=True,
    ):
        super().__init__(config, project_dir,
                         output_tags,
                         upload_dir,
                         download_resources, )

    def convert(self, input_data, output_data, format, is_dir=True, **kwargs):
        if isinstance(format, str):
            _format = AiCenterFormat.from_string(format)
        if _format == AiCenterFormat.PULC:
            self.convert_to_pulc(input_data, output_data, is_dir=is_dir)
        else:
            super().convert(input_data, output_data, format, is_dir=is_dir, **kwargs)

    def convert_to_pulc(self, input_data, output_dir, is_dir=True):
        ensure_dir(output_dir)
        output_file = os.path.join(output_dir, "result.txt")
        records = []
        item_iterator = self.iter_from_dir if is_dir else self.iter_from_json_file

        for item in item_iterator(input_data):
            records.append(f'{item["input"]["image"]}\t{prettify_result(item["output"]["choice"])}')

        with io.open(output_file, mode="w", encoding="utf8") as fout:
            fout.write("\n".join(records))

    def _get_supported_formats(self):
        is_mig = False
        if len(self._data_keys) > 1:
            return [
                Format.JSON.name,
                Format.JSON_MIN.name,
                Format.CSV.name,
                Format.TSV.name,
                AiCenterFormat.PULC.name,
            ]
        output_tag_types = set()
        input_tag_types = set()
        for info in self._schema.values():
            output_tag_types.add(info["type"])
            for input_tag in info["inputs"]:
                if input_tag.get("valueList"):
                    is_mig = True
                if input_tag["type"] == "Text" and input_tag.get("valueType") == "url":
                    logger.error('valueType="url" are not supported for text inputs')
                    continue
                input_tag_types.add(input_tag["type"])

        all_formats = [f.name for f in AiCenterFormat]
        if not ("Text" in input_tag_types and "Labels" in output_tag_types):
            all_formats.remove(Format.CONLL2003.name)
        if is_mig or not (
                "Image" in input_tag_types
                and (
                        "RectangleLabels" in output_tag_types
                        or "Rectangle" in output_tag_types
                        and "Labels" in output_tag_types
                )
        ):
            all_formats.remove(Format.VOC.name)
        if is_mig or not (
                "Image" in input_tag_types
                and (
                        "RectangleLabels" in output_tag_types
                        or "PolygonLabels" in output_tag_types
                )
                or "Rectangle" in output_tag_types
                and "Labels" in output_tag_types
                or "PolygonLabels" in output_tag_types
                and "Labels" in output_tag_types
        ):
            all_formats.remove(Format.COCO.name)
            all_formats.remove(Format.YOLO.name)
        if not (
                "Image" in input_tag_types
                and (
                        "BrushLabels" in output_tag_types
                        or "brushlabels" in output_tag_types
                        or "Brush" in output_tag_types
                        and "Labels" in output_tag_types
                )
        ):
            all_formats.remove(Format.BRUSH_TO_NUMPY.name)
            all_formats.remove(Format.BRUSH_TO_PNG.name)
        if not (
                ("Audio" in input_tag_types or "AudioPlus" in input_tag_types)
                and "TextArea" in output_tag_types
        ):
            all_formats.remove(Format.ASR_MANIFEST.name)
        if is_mig or ('Video' in input_tag_types and 'TimelineLabels' in output_tag_types):
            all_formats.remove(Format.YOLO_OBB.name)

        return all_formats
