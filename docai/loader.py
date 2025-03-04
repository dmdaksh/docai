from typing import Any, Dict, List

import torch
from loguru import logger
from PIL import Image

from docai.utils import read_json, train_data_format


class LayoutLMv3Loader(torch.utils.data.Dataset):
    """
    A dataset loader for LayoutLMv3 that reads training data from a JSON file,
    formats it, and applies a processor to generate tokenized encodings.
    """

    def __init__(self, json_path: str, processor: Any) -> None:
        """
        Initializes the loader with a JSON file path and a processor.

        Args:
            json_path (str): Path to the JSON file containing training data.
            processor (Any): A processor (e.g., LayoutLMv3Processor) to tokenize and encode images.
        """
        raw_data = read_json(json_path)
        # Ensure that raw_data is a list, as expected by train_data_format.
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        self.json_data: List[Dict[str, Any]] = train_data_format(raw_data)
        if processor is None:
            raise ValueError("Processor must be provided")
        self.processor = processor

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.json_data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves and processes a data sample at the specified index.

        Args:
            index (int): The index of the data sample.

        Returns:
            Dict[str, Any]: A dictionary with keys: "input_ids", "attention_mask", "bbox",
                            "pixel_values", and "labels".
        """
        data = self.json_data[index]

        try:
            img = Image.open(data["img_path"]).convert("RGB")
        except Exception as e:
            logger.error("Error opening image {}: {}", data.get("img_path"), e)
            raise

        words = data["tokens"]
        labels = data["ner_tag"]
        bboxes = data["bboxes"]

        encoding = self.processor(
            images=img,
            words=[words],
            boxes=[bboxes],
            word_labels=[labels],
            max_length=512,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"].squeeze(0),
        }
