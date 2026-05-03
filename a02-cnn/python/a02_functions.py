# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import time

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from a02_helper import DEVICE, accuracy, train_val_split


# %% [markdown]
# # Task 1: Mountain Climb CNN


# %%
class ClimbCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        # TODO: your code here
        # Convolution layer must be stored as `self.conv`.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: your code here
        return x


# %% [markdown]
# # Task 2: Simple CNN


# %%
class SimpleMLP(nn.Module):
    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.lin1 = nn.Linear(40, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x).relu()
        x = x + self.lin2(x).relu()
        x = self.lin3(x)
        return x


# %%


class SimpleCNN(nn.Module):
    def __init__(
        self,
        channels: int = 25,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        linear_in: int = 25,
    ) -> None:
        super().__init__()
        # TODO: your code here
        # Name each convolutional layer `self.conv1`, `self.conv2` etc.

        # use these attributes later for visualization (Task 3)
        self.store_embeddings = False
        self.embeddings: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension (of size 1)
        if len(x.shape) == 1:
            # unbatched input: [N] -> [1, N]
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            # batched input: [B, N] -> [B, 1, N]
            x = x.unsqueeze(1)

        # TODO: your code here

        # For task 3: store information about the forward pass in self.embeddings.
        # TODO: your code here

        return y


# %% [markdown]
# ### Training


# %%
def train_model(
    data: dict[str, np.ndarray],
    model: nn.Module,
    lr: float = 1e-2,
    batch_size: int = 64,
    epochs: int = 100,
    eval_every: int = 10,
) -> dict[str, list[float]]:
    # Split data into train and validation.
    x_train, x_val, y_train, y_val = train_val_split(data)

    # Create PyTorch dataset and data loader.
    # TODO: your code here
    train_dataset = ...
    train_loader = ...

    # Set up logging.
    results = {
        "train_losses": [],
        "val_losses": [],
        "test_losses": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }
    t0 = time.time()

    # Send model to accelerator (if available)
    model = model.to(DEVICE)

    # Define loss function and optimizer.
    # TODO: your code here
    loss_fn = ...  # (use cross-entropy loss)
    optimizer = ...  # (use, e.g., Adam)

    # Training loop.
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Forward pass: Compute the model's output and the loss. Store the
            # computed loss in the results dict (using loss.item()).
            # TODO: your code here
            output = ...
            loss = ...
            results["train_losses"].append(loss.item())

            # Backward pass: Compute the gradients of the loss with respect to all
            # the learnable parameters. Update the model's parameters using gradient
            # descent. Zero out the gradients for the next iteration.
            # TODO: your code here

        # Logging (no need to modify this)
        if epoch % eval_every == 0:
            results["train_acc"].append(accuracy(model, x_train, y_train))
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val)
            results["val_acc"].append(accuracy(model, x_val, y_val))
            results["val_losses"].append(val_loss.item())

            t1 = time.time()
            print(
                "epoch {}, dt {:.2f}s, train_loss {:.3e}, val_loss {:.3e}, train_acc {:.1f}, val_acc {:.1f}".format(
                    epoch,
                    t1 - t0,
                    loss.item(),
                    results["val_losses"][-1],
                    results["train_acc"][-1],
                    results["val_acc"][-1],
                )
            )
            t0 = t1

    # Final model assessment.
    x_test = torch.Tensor(data["x_test"]).to(torch.float).to(DEVICE)
    y_test = torch.LongTensor(data["y_test"]).to(DEVICE)
    with torch.no_grad():
        test_loss = loss_fn(model(x_test), y_test)
    results["test_acc"].append(accuracy(model, x_test, y_test))
    results["test_losses"].append(test_loss.item())

    print(
        f"Final result: "
        f"train_loss {results['train_losses'][-1]:.3e}, "
        f"val_loss {results['val_losses'][-1]:.3e}, "
        f"test_loss {results['test_losses'][-1]:.3e}, "
        f"train_acc {results['train_acc'][-1]:.1f}, "
        f"val_acc {results['val_acc'][-1]:.1f}, "
        f"test_acc {results['test_acc'][-1]:.1f}"
    )

    return results
