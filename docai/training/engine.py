import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_fn(
    data_loader: DataLoader, model: nn.Module, optimizer: Optimizer, device: str = "cpu"
) -> float:
    """
    Trains the model for one epoch.

    Args:
        data_loader (DataLoader): The data loader for training data.
        model (nn.Module): The model to train.
        optimizer (Optimizer): The optimizer for updating the model parameters.
        device (str, optional): The device to train on, e.g., 'cpu' or 'cuda'. Defaults to "cpu".

    Returns:
        float: The average loss over the training epoch.
    """
    device_obj = torch.device(device)
    model.train()
    total_loss: float = 0.0

    for batch in tqdm(data_loader, total=len(data_loader), desc="Training"):
        # Move data to the specified device
        batch = {k: v.to(device_obj) for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward pass and compute loss. Expecting model to return (output, loss)
        _, loss = model(**batch)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader: DataLoader, model: nn.Module, device: str = "cpu") -> float:
    """
    Evaluates the model on the validation or test set.

    Args:
        data_loader (DataLoader): The data loader for validation or test data.
        model (nn.Module): The model to evaluate.
        device (str, optional): The device to evaluate on, e.g., 'cpu' or 'cuda'. Defaults to "cpu".

    Returns:
        float: The average loss over the evaluation.
    """
    device_obj = torch.device(device)
    model.eval()
    total_loss: float = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
            # Move data to the specified device
            batch = {k: v.to(device_obj) for k, v in batch.items()}

            # Forward pass and compute loss
            _, loss = model(**batch)
            total_loss += loss.item()

    return total_loss / len(data_loader)
