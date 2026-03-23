"""Training loop (M45: config loss, M46: fallback metric, M49: sklearn metrics)."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

logger = logging.getLogger(__name__)

LOSS_REGISTRY = {
    "mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.SmoothL1Loss,
    "bce": nn.BCEWithLogitsLoss, "ce": nn.CrossEntropyLoss,
}


class Trainer:
    """Model trainer with config-based loss and sklearn evaluation."""

    def __init__(self, config: Dict):
        self.config = config
        # M45: loss function from config, not hardcoded MSELoss
        loss_name = config.get("loss_function", "mse")
        self.criterion = LOSS_REGISTRY.get(loss_name, nn.MSELoss)()
        self.early_stop_metric = config.get("early_stop_metric", "val_loss")
        # M51: multi-task loss weighting from config
        self.task_weights = config.get("task_weights", {"primary": 1.0})

    def train(self, model: nn.Module, train_loader, val_loader=None,
              n_epochs: int = 100, lr: float = 1e-3, patience: int = 10,
              device: str = "cpu") -> Dict:
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        best_metric = float("inf")
        best_state = None
        wait = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                pred = model(batch.to(device))
                loss = self.criterion(pred.squeeze(), batch.y.float().to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            history["train_loss"].append(epoch_loss / max(len(train_loader), 1))

            # Validation with sklearn metrics (M49)
            if val_loader:
                val_results = self.validate(model, val_loader, device)
                history["val_loss"].append(val_results.get("loss", float("inf")))

                # M46: fallback when early stopping metric not in results
                metric_val = val_results.get(self.early_stop_metric,
                                              val_results.get("loss", float("inf")))

                if metric_val < best_metric:
                    best_metric = metric_val
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        if best_state:
            model.load_state_dict(best_state)
        return history

    def validate(self, model: nn.Module, loader, device: str = "cpu") -> Dict:
        """Validate using sklearn metrics directly (M49)."""
        model.eval()
        all_preds, all_labels = [], []
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                pred = model(batch.to(device))
                labels = batch.y.float().to(device)
                loss = self.criterion(pred.squeeze(), labels)
                total_loss += loss.item()
                all_preds.extend(pred.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        # M49: sklearn metrics directly
        results = {"loss": total_loss / max(len(loader), 1)}
        try:
            results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            results["r2"] = float(r2_score(y_true, y_pred))
        except Exception:
            pass
        return results
