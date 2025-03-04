import json
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pytesseract
from matplotlib.patches import Rectangle
from PIL import Image


def read_json(json_path: str) -> Dict[str, Any]:
    """
    Read a JSON file and return its content as a dictionary.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON data.
    """
    with open(json_path, "r") as fp:
        return json.load(fp)


def train_data_format(json_to_dict: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert JSON data to the format required for training.

    Args:
        json_to_dict (List[Dict[str, Any]]): List of data items from the JSON file.

    Returns:
        List[Dict[str, Any]]: List of formatted data dictionaries.
    """
    final_list: List[Dict[str, Any]] = []
    for count, item in enumerate(json_to_dict, start=1):
        test_dict = {
            "id": count,
            "img_path": item["file_name"],
            "tokens": [cont["text"] for cont in item["annotations"]],
            "bboxes": [cont["box"] for cont in item["annotations"]],
            "ner_tag": [cont["label"] for cont in item["annotations"]],
        }
        final_list.append(test_dict)
    return final_list


def scale_bounding_box(box: List[int], width: float, height: float) -> List[int]:
    """
    Scale bounding box coordinates to a percentage of the image size.

    Args:
        box (List[int]): Bounding box coordinates [x1, y1, x2, y2].
        width (float): Image width.
        height (float): Image height.

    Returns:
        List[int]: Scaled bounding box coordinates.
    """
    return [
        int(100 * box[0] / width),
        int(100 * box[1] / height),
        int(100 * box[2] / width),
        int(100 * box[3] / height),
    ]


def process_bbox(box: List[int]) -> List[int]:
    """
    Convert bounding box from [x1, y1, x2, y2] to [x1, y1, width, height].

    Args:
        box (List[int]): Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        List[int]: Converted bounding box in [x1, y1, width, height] format.
    """
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def dataSetFormat(img_file: Image.Image) -> Tuple[Dict[str, Any], int, int]:
    """
    Extract text and bounding box information from an image.

    Args:
        img_file (Image.Image): The input image.

    Returns:
        Tuple[Dict[str, Any], int, int]: A tuple containing a dictionary with extracted tokens
                                         and bounding boxes, and the image width and height.
    """
    width, height = img_file.size

    # Perform OCR using pytesseract
    data = pytesseract.image_to_data(img_file, output_type=pytesseract.Output.DICT)

    test_dict: Dict[str, Any] = {"tokens": [], "bboxes": []}
    test_dict["img_path"] = img_file

    for i in range(len(data["text"])):
        if not data["text"][i].strip():
            continue

        test_dict["tokens"].append(data["text"][i])
        processed_bbox = process_bbox(
            [
                data["left"][i],
                data["top"][i],
                data["left"][i] + data["width"][i],
                data["top"][i] + data["height"][i],
            ]
        )
        scaled_bbox = scale_bounding_box(processed_bbox, width, height)
        test_dict["bboxes"].append(scaled_bbox)

    return test_dict, width, height


def plot_img(
    image: Image.Image,
    bbox_list: List[List[int]],
    label_list: List[Any],
    prob_list: List[float],
    width: float,
    height: float,
    dpi: int = 300,
) -> None:
    """
    Plot image with bounding boxes and labels.

    Args:
        image (Image.Image): The image to be plotted.
        bbox_list (List[List[int]]): List of bounding boxes.
        label_list (List[Any]): List of labels corresponding to bounding boxes.
        prob_list (List[float]): List of probabilities corresponding to bounding boxes.
        width (float): Image width.
        height (float): Image height.
        dpi (int, optional): DPI for saving the image. Defaults to 300.
    """
    plt.figure(figsize=(10, 10), dpi=dpi)
    plt.imshow(image)
    ax = plt.gca()

    for i, bbox in enumerate(bbox_list):
        # Convert percentage-based bbox back to pixel coordinates
        rect = Rectangle(
            (bbox[0] * width / 100, bbox[1] * height / 100),
            bbox[2],
            bbox[3],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            bbox[0] * width / 100,
            bbox[1] * height / 100,
            f"{label_list[i]}: {prob_list[i]:.2f}",
            bbox={"facecolor": [1, 1, 1], "alpha": 0.5},
            clip_box=ax.clipbox,
            clip_on=True,
        )

    plt.savefig("test_image.jpg", dpi=dpi)
    plt.show()
    plt.clf()

