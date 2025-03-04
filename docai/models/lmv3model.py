from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import LayoutLMv3ForTokenClassification


class ModelModule(nn.Module):
    """
    A model module that extends a pretrained LayoutLMv3 for token classification
    with an additional classification layer.
    """

    def __init__(self, n_classes: int) -> None:
        """
        Initializes the ModelModule.

        Args:
            n_classes (int): The number of classes for the token classification.
        """
        super().__init__()
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            "models_inputs/layoutlmv3"
        )
        self.cls_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the tokens.
            bbox (torch.Tensor): Bounding box coordinates.
            pixel_values (torch.Tensor): Pixel values of the image.
            labels (Optional[torch.Tensor]): Ground truth labels.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the logits and, if provided, the computed loss.
        """
        # Pass inputs through the pretrained LayoutLMv3 model.
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )

        # Apply the classification layer to the logits.
        logits = self.cls_layer(output.logits)

        # Compute probabilities.
        probabilities = F.softmax(logits, dim=1)

        # Get the top prediction and its probability.
        top_p, top_class = probabilities.topk(1, dim=1)
        logger.debug("Probability score: {}", probabilities)
        logger.debug("Top prediction and class: {} {}", top_p, top_class)

        # Calculate loss if labels are provided.
        loss = self.loss_fn(logits, labels) if labels is not None else None

        return logits, loss

    @staticmethod
    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross entropy loss between predictions and target labels.

        Args:
            pred (torch.Tensor): Logits from the model.
            target (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: The computed cross entropy loss.
        """
        return nn.CrossEntropyLoss()(pred.view(-1, pred.size(-1)), target.view(-1))

