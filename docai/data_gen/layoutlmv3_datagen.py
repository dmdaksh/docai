import json
from typing import Any, Dict, List

import typer
from loguru import logger

app = typer.Typer()


def convert_bounding_box(
    x: float, y: float, width: float, height: float
) -> List[float]:
    """
    Converts the given bounding box coordinates to (x1, y1, x2, y2) format.

    Args:
        x (float): The x-coordinate of the top-left corner of the bounding box.
        y (float): The y-coordinate of the top-left corner of the bounding box.
        width (float): The width of the bounding box.
        height (float): The height of the bounding box.

    Returns:
        List[float]: A list of four coordinates [x1, y1, x2, y2].
    """
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height
    return [x1, y1, x2, y2]


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads JSON data from the specified file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        List[Dict[str, Any]]: The loaded JSON data.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            logger.info(f"Loaded data from {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in file {file_path}: {e}")
        return []


def process_annotated_image(annotated_image: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single annotated image entry and converts it into the desired format.

    Args:
        annotated_image (Dict[str, Any]): The annotated image data.

    Returns:
        Dict[str, Any]: The processed data for the image.
    """
    processed_data: Dict[str, Any] = {}
    annotations: List[Dict[str, Any]] = []

    # Check if the annotated_image has sufficient data
    if len(annotated_image) < 8:
        logger.debug("Annotated image skipped due to insufficient data.")
        return {}

    # Process file information
    ocr_value = annotated_image.get("ocr", "")
    if ocr_value:
        filename = ocr_value.split("8081/")[-1]
        logger.debug(f"Processing file: {filename}")
        processed_data["file_name"] = f"cleaned_eobs_img/{filename}"
    else:
        logger.warning("Missing 'ocr' key in annotated_image.")

    # Process bounding box dimensions (using the first bbox entry for image dimensions)
    bbox_info = annotated_image.get("bbox", [])
    if (
        bbox_info
        and isinstance(bbox_info, list)
        and "original_width" in bbox_info[0]
        and "original_height" in bbox_info[0]
    ):
        original_width = bbox_info[0]["original_width"]
        original_height = bbox_info[0]["original_height"]
        processed_data["width"] = original_width
        processed_data["height"] = original_height
    else:
        logger.warning("Missing or invalid 'bbox' information in annotated_image.")

    # Process annotations
    bounding_boxes = annotated_image.get("bbox", [])
    transcriptions = annotated_image.get("transcription", [])
    labels = annotated_image.get("label", [])

    for bbox, transcription, label in zip(bounding_boxes, transcriptions, labels):
        ann_dict: Dict[str, Any] = {}
        text = transcription
        logger.debug(f"Text: {text}")

        box = convert_bounding_box(
            x=bbox.get("x", 0),
            y=bbox.get("y", 0),
            width=bbox.get("width", 0),
            height=bbox.get("height", 0),
        )
        ann_dict["box"] = box
        ann_dict["text"] = text
        # If label has a "labels" key, take the last element; otherwise, use None.
        ann_dict["label"] = label.get("labels", [])[-1] if label.get("labels") else None
        annotations.append(ann_dict)

    if annotations:
        processed_data["annotations"] = annotations

    return processed_data


def process_data(input_file: str, output_file: str) -> None:
    """
    Processes the input JSON data and writes the transformed data to the output file.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    input_data = load_json_data(input_file)
    output_data: List[Dict[str, Any]] = []

    for annotated_image in input_data:
        processed_image = process_annotated_image(annotated_image)
        if processed_image:
            output_data.append(processed_image)

    try:
        with open(output_file, "w") as file:
            json.dump(output_data, file, indent=4)
            logger.info(f"Processed data written to {output_file}")
    except IOError as e:
        logger.error(f"Failed to write to {output_file}: {e}")


@app.command()
def main(
    input_file: str = typer.Argument(..., help="Path to the input JSON file"),
    output_file: str = typer.Argument(..., help="Path to the output JSON file"),
) -> None:
    """
    CLI entry point to process annotated images.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file.
    """
    process_data(input_file, output_file)


if __name__ == "__main__":
    app()

