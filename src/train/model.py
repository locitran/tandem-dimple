import os
from tqdm import tqdm  # Import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data import get_r20000_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
lr=5e-5
epochs = 1000
dropout_rate = 0
l2 = 1e-4
batch_size = 300
patience = 50

class DNN(nn.Module):

    def __init__(self, input_dim=33, hidden_dims=(33, 33, 33, 33, 10), output_dim=2, dropout_rate=0) -> None:
        super().__init__()

        layers = []
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)

            total_loss += loss.item() * x.size(0)
            total_correct += (pred == y).sum().item()
            total_samples += x.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train(feat_names, input_dim=50, folder="r20000_DNN"):
    dataloaders = get_r20000_dataloaders(
        feat_names=feat_names,
        input_dim=input_dim,
        batch_size=batch_size,
    )

    model = DNN(input_dim=input_dim, hidden_dims=(33, 33, 33, 33, 10), output_dim=2, dropout_rate=dropout_rate)
    model = model.to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=l2)
    criterion = nn.CrossEntropyLoss()

    output_dir = os.path.join("runs", folder)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    writer = SummaryWriter(output_dir)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Training on {device}...")
    pbar = tqdm(range(1, epochs + 1), desc="Epochs")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        train_total = 0

        for batch in dataloaders["train"]:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_total += x.size(0)

        train_loss /= train_total

        val_metrics = evaluate(model, dataloaders["val"], criterion)
        val_loss = val_metrics["loss"]

        writer.add_scalars(
            "loss",
            {
                "train": train_loss,
                "val": val_loss,
            },
            epoch,
        )

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        pbar.set_postfix(train_loss=f"{train_loss:.4f}",val_loss=f"{val_loss:.4f}",patience=f"{epochs_without_improvement}/{patience}",)

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, dataloaders["test"], criterion)
    writer.add_scalar("test/loss", test_metrics["loss"], best_epoch)
    writer.add_scalar("test/accuracy", test_metrics["accuracy"], best_epoch)
    writer.close()

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy'] * 100:.2f}%")

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "checkpoint": checkpoint_path,
    }

if __name__ == "__main__":
    import sys

    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from src.features import TANDEM_FEATS, dynamics_feat, structure_feat, sequence_feat, rhapsody_feat, evolution_feat, chemical_feat

    evo = list(evolution_feat.keys())
    evo_str_dyn = list(evolution_feat.keys()) + list(structure_feat.keys()) + list(dynamics_feat.keys())
    
    train(feat_names=evo, folder='evo')
    train(feat_names=evo_str_dyn, folder='evo_str_dyn')
