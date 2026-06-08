import os
import json
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm  # Import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
try:
    from .data import get_r20000_dataloaders
    from .tools import set_seed, log_message, plot_training_history
except ImportError:
    from data import get_r20000_dataloaders
    from tools import set_seed, log_message, plot_training_history

seed = 17
device = "cuda" if torch.cuda.is_available() else "cpu"
lr=5e-5
epochs = 10000
dropout_rate = 0.4
l2 = 1e-4
batch_size = 300
patience = 1000


class DNN(nn.Module):

    def __init__(self, input_dim=33, hidden_dims=(33, 33, 33, 33, 10), output_dim=2, dropout_rate=0):
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


def get_class_weights(dataloader):
    labels = dataloader.dataset.y
    counts = torch.bincount(labels, minlength=2).float()
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return weights


def train(feat_names, input_dim=50, folder="r20000_DNN"):
    set_seed(seed)

    dataloaders = get_r20000_dataloaders(feat_names=feat_names,input_dim=input_dim,batch_size=batch_size,)
    model = DNN(input_dim=input_dim, hidden_dims=(33, 33, 33, 33, 10), output_dim=2, dropout_rate=dropout_rate)
    model = model.to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=l2)

    output_dir = os.path.join("runs", folder)
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    log_file = os.path.join(output_dir, "log.txt")
    config_file = os.path.join(output_dir, "train_config.json")
    summary_file = os.path.join(output_dir, "training_summary.json")
    writer = SummaryWriter(output_dir)

    with open(log_file, "w") as f:
        f.write("")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"epoch": [],"train_loss": [],"val_loss": [],}

    n_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    class_weights = get_class_weights(dataloaders["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_config = {
        "folder": folder,
        "feat_names": list(feat_names),
        "input_dim": input_dim,
        "hidden_dims": [33, 33, 33, 33, 10],
        "output_dim": 2,
        "dropout_rate": dropout_rate,
        "learning_rate": lr,
        "l2": l2,
        "batch_size": batch_size,
        "patience": patience,
        "epochs": epochs,
        "seed": seed,
        "class_weights": [class_weights[0].item(), class_weights[1].item()],
        "n_parameters": n_parameters,
    }
    with open(config_file, "w") as f:
        json.dump(train_config, f, indent=2)

    log_message(f"Trainable parameters: {n_parameters:,}", log_file)
    log_message(f"Class weights: class0={class_weights[0].item():.4f}, class1={class_weights[1].item():.4f}", log_file)
    pbar = tqdm(range(1, epochs + 1), desc="Epochs")
    for epoch in pbar:
        
        # Fit training data and calculate training loss
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

        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for batch in dataloaders["val"]:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_total += x.size(0)
        val_loss /= val_total

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss,}, epoch,)
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
                    "train_config": train_config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        pbar.set_postfix(train_loss=f"{train_loss:.4f}",val_loss=f"{val_loss:.4f}",patience=f"{epochs_without_improvement}/{patience}",)
        if epochs_without_improvement >= patience:
            log_message(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.", log_file)
            break

    figure_path = plot_training_history(history, output_dir)
    output = {
        "folder": folder,
        "input_dim": input_dim,
        "n_features": len(feat_names),
        "n_parameters": n_parameters,
        "class0_weight": class_weights[0].item(),
        "class1_weight": class_weights[1].item(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": checkpoint_path,
        "loss_curve": figure_path,
    }
    with open(summary_file, "w") as f:
        json.dump(output, f, indent=2)
    writer.close()
    log_message(f"Best epoch: {best_epoch}", log_file)
    log_message(f"Best validation loss: {best_val_loss:.4f}", log_file)
    log_message(f"Checkpoint: {checkpoint_path}", log_file)
    log_message(f"Training summary: {summary_file}", log_file)
    return output


if __name__ == "__main__":
    import sys
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, package_root)
    from src.features import TANDEM_FEATS, dynamics_feat, structure_feat, sequence_feat, rhapsody_feat, evolution_feat, chemical_feat

    psid = ["wtPSIC", "deltaPSIC"]
    chem = list(chemical_feat.keys()) 
    evo = list(evolution_feat.keys())
    evo_str_dyn = list(evolution_feat.keys()) + list(structure_feat.keys()) + list(dynamics_feat.keys())
    str_dyn = list(structure_feat.keys()) + list(dynamics_feat.keys())
    psid_str_dyn = psid + list(structure_feat.keys()) + list(dynamics_feat.keys())
    
    seq_chem = list(sequence_feat.keys())
    seq = list(evolution_feat.keys())


    seq_chem_str_dyn = list(sequence_feat.keys()) + list(structure_feat.keys()) + list(dynamics_feat.keys())

    train(feat_names=seq_chem, folder='seq_chem')
    train(feat_names=seq, folder='seq_chem')
    train(feat_names=seq_chem_str_dyn, folder='seq_chem_str_dyn')
    train(feat_names=str_dyn, folder='str_dyn')


    train(feat_names=chem, folder='chem')
    train(feat_names=psid, folder='psid')
    train(feat_names=evo, folder='evo')
    train(feat_names=evo_str_dyn, folder='evo_str_dyn')
    train(feat_names=str_dyn, folder='str_dyn')
    train(feat_names=psid_str_dyn, folder='psid_str_dyn')
    
    
    train(feat_names=TANDEM_FEATS['v1.1'], folder='tandem-33')
