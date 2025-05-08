import copy
from pathlib import Path
from typing import Any, Dict, Optional

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cs410_han.evaluate import evaluate_model
from cs410_han.logger import logger


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    count = 0

    if len(data_loader) == 0:
        logger.warning(f"Empty data loader at epoch {epoch + 1}.")
        return 0.0

    total_batches = len(data_loader)
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        TextColumn("Loss: {task.fields[loss]:.4f}"),
        expand=True,
    )
    task = progress.add_task(
        f"[green]Epoch {epoch + 1}/{num_epochs}", total=total_batches, loss=float("inf")
    )

    with progress:
        for batch_idx, batch in enumerate(data_loader):
            # move batch to device
            docs, labels, sent_lengths, doc_lengths = [b.to(device) for b in batch]
            # skip invalid inputs
            if torch.any(doc_lengths <= 0) or docs.nelement() == 0:
                progress.update(
                    task, advance=1, loss=total_loss / count if count else 0.0
                )
                continue

            optimizer.zero_grad()
            outputs, _, _ = model(docs, sent_lengths, doc_lengths)
            if outputs is None:
                progress.update(
                    task, advance=1, loss=total_loss / count if count else 0.0
                )
                continue

            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                logger.error(f"NaN loss at epoch {epoch + 1}, batch {batch_idx + 1}.")
                progress.update(task, advance=1, loss=float("nan"))
                continue

            loss.backward()
            # gradient clipping for protection against exploding gradients
            # enable if necessary
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            val = loss.item()
            total_loss += val
            count += 1
            progress.update(task, advance=1, loss=val)

    avg_loss = total_loss / count if count else 0.0
    if count < total_batches:
        logger.warning(
            f"{count}/{total_batches} batches processed at epoch {epoch + 1}."
        )
    return avg_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    momentum: float,
    num_epochs: int,
    patience: int,
    model_save_dir: Path,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Train model with early stopping."""
    logger.info(f"Training up to {num_epochs} epochs.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_accuracy = 0.0
    epochs_no_improve = 0
    best_state: Optional[Dict[str, Any]] = None
    save_path = model_save_dir / "han_model_best.pt"

    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs
        )
        val_loss, val_acc = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            f"Epoch {epoch + 1}/{num_epochs} Validation",
        )
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}: train={train_loss:.4f}, "
            f"val={val_loss:.4f}, acc={val_acc:.2f}%"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            epochs_no_improve = 0
            try:
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, save_path)
                logger.success(f"Saved best model to {save_path} at epoch {epoch + 1}.")
            except Exception as e:
                logger.error(f"Save failed at epoch {epoch + 1}: {e}")
                best_state = None
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                logger.warning(f"Stopping early at epoch {epoch + 1}.")
                break

    if best_state:
        logger.success(f"Best accuracy: {best_accuracy:.2f}%")
    else:
        logger.warning("No improvement observed.")

    return best_state
