import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_arrays(model, x, y, criterion):
    model.eval()

    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float32).to(device)
        y = torch.as_tensor(y, dtype=torch.long).to(device)
        logits = model(x)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)

    y_prob = prob.cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_true = y.cpu().numpy()
    total_samples = len(y_true)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (true_positive + true_negative) / total_samples
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    auc = float("nan")
    auprc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "auprc": auprc,
    }


def evaluate(model, dataloader, criterion):
    xs = []
    ys = []
    for batch in dataloader:
        xs.append(batch["x"])
        ys.append(batch["y"])

    x = torch.cat(xs)
    y = torch.cat(ys)
    return evaluate_arrays(model, x, y, criterion)
