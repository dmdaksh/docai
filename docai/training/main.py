import numpy as np
import torch
import typer
from loguru import logger
from torch.optim import AdamW
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

from docai.training.engine import eval_fn, train_fn
from docai.datasets.loader import (
    LayoutLMv3Loader as Loader,
)

app = typer.Typer()


@app.command()
def main(
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    training_json: str = "layoutlmv3_training.json",
    model_save_path: str = "./model.bin",
) -> None:
    # Determine device as a string and create torch.device for other uses.
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device_str}")

    # Initialize processor and model.
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        "models_inputs/layoutlmv3", ignore_mismatched_sizes=True
    )
    processor = LayoutLMv3Processor(
        tokenizer=tokenizer, feature_extractor=feature_extractor
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained("models_inputs/layoutlmv3")

    # Move model to device. We pass the device as a string here.
    model = model.to(device_str)  # type: ignore
    logger.info("Processor and model initialized.")

    # Prepare dataset and dataloader.
    dataset = Loader(training_json, processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    logger.info(f"Dataset loaded with {len(dataset)} samples.")

    # Define optimizer.
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_loss = np.inf
    loss_list = []

    # Training and evaluation loop.
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # Training.
        train_loss = train_fn(dataloader, model, optimizer, device_str)
        logger.info(f"Epoch {epoch + 1}: Training loss = {train_loss:.4f}")

        if train_loss < best_loss:
            torch.save(model.state_dict(), model_save_path)
            best_loss = train_loss
            logger.info(
                f"Best model saved at epoch {epoch + 1} with loss {best_loss:.4f}"
            )

        if epoch % 10 == 0:
            periodic_save_path = f"./model_{epoch + 1}.bin"
            torch.save(model.state_dict(), periodic_save_path)
            logger.info(f"Periodic model saved at {periodic_save_path}")

        # Evaluation.
        eval_loss = eval_fn(dataloader, model, device_str)
        logger.info(f"Epoch {epoch + 1}: Evaluation loss = {eval_loss:.4f}")
        loss_list.append(train_loss)

    np.save("loss_list.npy", np.array(loss_list))
    logger.info("Training complete. Loss list saved to loss_list.npy.")


if __name__ == "__main__":
    app()

