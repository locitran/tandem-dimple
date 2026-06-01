import os
from tqdm import tqdm  # Import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data import get_r20000_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
lr=5e-5
epochs = 300
output_dir="runs/r20000_resnet18"
class MLPBlock(nn.Module):
    """
    MLP version of ResNet BasicBlock.

    Original ResNet BasicBlock:
        Conv -> BN -> ReLU -> Conv -> BN -> skip-add -> ReLU

    MLP version:
        Linear -> BN -> ReLU -> Linear -> BN -> skip-add -> ReLU
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

        # If input and output dimensions differ, project input to out_dim
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.main(x)
        out = out + residual
        out = self.activation(out)
        return out


class MLPResNet18(nn.Module):
    """
    MLP version of ResNet-18.

    ResNet-18 block structure:
        [2, 2, 2, 2]

    For vector input:
        input_dim -> hidden_dims[0] -> hidden_dims[1] -> hidden_dims[2] -> hidden_dims[3]
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(64, 128, 256, 512),
        num_blocks=(2, 2, 2, 2),
        dropout=0.0,
    ):
        super().__init__()

        assert len(hidden_dims) == 4
        assert len(num_blocks) == 4

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(
            in_dim=hidden_dims[0],
            out_dim=hidden_dims[0],
            n_blocks=num_blocks[0],
            dropout=dropout,
        )

        self.stage2 = self._make_stage(
            in_dim=hidden_dims[0],
            out_dim=hidden_dims[1],
            n_blocks=num_blocks[1],
            dropout=dropout,
        )

        self.stage3 = self._make_stage(
            in_dim=hidden_dims[1],
            out_dim=hidden_dims[2],
            n_blocks=num_blocks[2],
            dropout=dropout,
        )

        self.stage4 = self._make_stage(
            in_dim=hidden_dims[2],
            out_dim=hidden_dims[3],
            n_blocks=num_blocks[3],
            dropout=dropout,
        )

        self.output_layer = nn.Linear(hidden_dims[3], output_dim)

    def _make_stage(self, in_dim, out_dim, n_blocks, dropout):
        layers = []

        # First block may change dimension
        layers.append(MLPBlock(in_dim, out_dim, dropout=dropout))

        # Remaining blocks keep same dimension
        for _ in range(1, n_blocks):
            layers.append(MLPBlock(out_dim, out_dim, dropout=dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.output_layer(x)
        return x

def train():
    """Train MLPResNet18 on the first R20000 fold."""
    dataloaders = get_r20000_dataloaders()
    sample_batch = next(iter(dataloaders["train"]))
    input_dim = sample_batch["x"].shape[1]
    model = MLPResNet18(input_dim=input_dim, output_dim=2, hidden_dims=(64, 16, 8, 4))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    writer = SummaryWriter(output_dir)

    best_val_loss = float("inf")
    best_epoch = 0

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

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)


        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )

        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    writer.close()

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint": checkpoint_path,
    }
    


if __name__ == "__main__":
    train()

    # batch_size = 32
    # input_dim = 70
    # output_dim = 2

    # model = MLPResNet18(
    #     input_dim=input_dim,
    #     output_dim=output_dim,
    #     hidden_dims=(64, 128, 256, 512),
    #     dropout=0.1,
    # )

    # x = torch.randn(batch_size, input_dim)
    # y = model(x)

    # print(model)
    # print("Input shape:", x.shape)
    # print("Output shape:", y.shape)

    # model = MLPResNet18(input_dim=33, output_dim=2)

    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(n_params)
