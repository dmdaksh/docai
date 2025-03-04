from typing import Optional, Tuple, cast

import numpy as np
import torch
import typer
from loguru import logger
from PIL import Image
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)
from transformers.tokenization_utils_base import BatchEncoding

from docai.lmv3model import ModelModule  # Consider explicit imports for clarity.
from docai.utils import (
    dataSetFormat,
    plot_img,
)

app = typer.Typer()


def initialize_processor() -> LayoutLMv3Processor:
    """
    Initializes the LayoutLMv3 processor with feature extractor and tokenizer.

    Returns:
        LayoutLMv3Processor: The initialized processor.
    """
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        "models_inputs/layoutlmv3", ignore_mismatched_sizes=True
    )
    processor = LayoutLMv3Processor(
        tokenizer=tokenizer, feature_extractor=feature_extractor
    )
    return processor


def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Loads an image from the specified path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Optional[Image.Image]: The loaded image, or None if an error occurs.
    """
    try:
        image = Image.open(image_path)
        image.show()  # Note: remove or disable in headless environments.
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None


def process_image(
    image: Image.Image, processor: LayoutLMv3Processor
) -> Tuple[BatchEncoding, float, float]:
    """
    Processes the image and prepares it for the LayoutLMv3 model.

    Args:
        image (Image.Image): The image to be processed.
        processor (LayoutLMv3Processor): The processor for LayoutLMv3.

    Returns:
        Tuple[BatchEncoding, float, float]: A tuple containing the processed image data,
                                              width scaling factor, and height scaling factor.
    """
    # dataSetFormat is assumed to be imported from another module.
    test_dict, width_scale, height_scale = dataSetFormat(image)

    encoding = processor(
        test_dict["img_path"].convert("RGB"),
        test_dict["tokens"],
        boxes=test_dict["bboxes"],
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    return encoding, width_scale, height_scale


def load_model(model_path: str, num_classes: int = 5) -> torch.nn.Module:
    """
    Loads the trained LayoutLMv3 model.

    Args:
        model_path (str): Path to the saved model file.
        num_classes (int): The number of classes for the model.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = ModelModule(num_classes)
    model.load_state_dict(torch.load(model_path))
    return model


def run_inference(
    model: torch.nn.Module, encoding: BatchEncoding, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs inference on the given model and encoding.

    Args:
        model (torch.nn.Module): The trained model.
        encoding (BatchEncoding): The processed image data.
        device (torch.device): The device to run the inference on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The predicted class labels,
                                                           prediction probabilities, and bounding boxes.
    """
    model.to(device)
    with torch.no_grad():
        input_ids = cast(torch.Tensor, encoding["input_ids"]).to(device).squeeze(0)
        attention_mask = (
            cast(torch.Tensor, encoding["attention_mask"]).to(device).squeeze(0)
        )
        bbox = cast(torch.Tensor, encoding["bbox"]).to(device).squeeze(0)
        pixel_values = (
            cast(torch.Tensor, encoding["pixel_values"]).to(device).squeeze(0)
        )

        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            bbox=bbox.unsqueeze(0),
            pixel_values=pixel_values.unsqueeze(0),
        )

        predictions = outputs.argmax(-1).squeeze().tolist()
        prob = torch.nn.functional.softmax(outputs, dim=1).squeeze()
        # Convert probabilities to CPU numpy array for normalization.
        prob_np = prob.cpu().numpy()
        normalized_prob = prob_np / np.sum(prob_np, axis=1, keepdims=True)
        output_prob = np.max(normalized_prob, axis=1)

    return torch.tensor(predictions), torch.tensor(output_prob), bbox


def filter_predictions(
    predictions: torch.Tensor,
    output_prob: torch.Tensor,
    bbox: torch.Tensor,
    encoding: BatchEncoding,
) -> torch.Tensor:
    """
    Filters and processes the predictions to remove subwords.

    Args:
        predictions (torch.Tensor): The predicted class labels.
        output_prob (torch.Tensor): The prediction probabilities.
        bbox (torch.Tensor): The bounding boxes.
        encoding (BatchEncoding): The processed image data.

    Returns:
        torch.Tensor: The filtered tensor containing boxes, predictions, and probabilities.
    """
    offset_mapping = cast(torch.Tensor, encoding["offset_mapping"]).squeeze()
    offset_array = np.array(offset_mapping.tolist())
    is_subword = offset_array[:, 0] != 0
    mask = ~torch.tensor(is_subword)

    true_predictions = predictions[mask]
    true_prob = output_prob[mask]
    true_boxes = bbox[mask]

    concat_torch = torch.column_stack((true_boxes, true_predictions, true_prob))

    # Filter predictions into five classes.
    # Assumes columns: 0-3: bounding box, 4: predicted class, 5: probability.
    final_tensor = torch.vstack(
        [
            concat_torch[
                (concat_torch[:, 4] == i)
                & (concat_torch[:, 3] == 0)
                & (concat_torch[:, 2] == 0)
            ]
            for i in range(1, 6)
        ]
    )

    return final_tensor


@app.command()
def main(
    image_path: str = typer.Option(..., help="Path to the input image"),
    model_path: str = typer.Option(..., help="Path to the saved model"),
) -> None:
    """
    Runs the entire inference pipeline using LayoutLMv3.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the saved model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    processor = initialize_processor()
    image = load_image(image_path)
    if image is None:
        logger.error("Image could not be loaded, exiting.")
        raise typer.Exit(code=1)

    encoding, width_scale, height_scale = process_image(image, processor)
    model = load_model(model_path)

    predictions, output_prob, bbox = run_inference(model, encoding, device)
    final_tensor = filter_predictions(predictions, output_prob, bbox, encoding)

    # Convert bbox tensor to list for plotting.
    bbox_list = final_tensor[:, :4].tolist()
    # Use the original image (converted to RGB) for plotting.
    plot_img(
        image.convert("RGB"),
        bbox_list,
        final_tensor[:, 4].tolist(),
        final_tensor[:, 5].tolist(),
        width_scale,
        height_scale,
    )


if __name__ == "__main__":
    app()

