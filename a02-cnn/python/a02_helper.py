import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mnist1d import get_dataset_args, get_templates, set_seed
from mnist1d.data import make_dataset
from numpy import ndarray
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 200)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup plotting
from IPython import get_ipython
import psutil

inTerminal = "IPKernelApp" not in get_ipython().config
inJupyterNb = any(
    filter(
        lambda x: x.endswith("jupyter-notebook"), psutil.Process().parent().cmdline()
    )
)
get_ipython().run_line_magic(
    "matplotlib", "" if inTerminal else "notebook" if inJupyterNb else "widget"
)


def nextplot():
    if inTerminal:
        plt.clf()  # this clears the current plot
    else:
        plt.figure()  # this creates a new plot


# ===
# Dataset
# ===


def get_raw_data() -> dict[str, np.ndarray]:
    set_seed(0)
    defaults = get_dataset_args()
    return make_dataset(defaults)


def shuffle_pixels(dataset: dict[str, np.ndarray]) -> dict[str, ndarray]:
    np.random.seed(0)
    shuffled_order = np.random.permutation(dataset["x"].shape[1])

    shuffled_dataset = dataset.copy()
    shuffled_dataset["x"] = shuffled_dataset["x"][:, shuffled_order]
    shuffled_dataset["x_test"] = shuffled_dataset["x_test"][:, shuffled_order]

    return shuffled_dataset


# ===
# Model Parameters
# ===


def count_model_params(model: nn.Module) -> int:
    with torch.no_grad():
        return sum([p.view(-1).shape[0] for p in model.parameters()])


# ===
# Plotting
# ===


def plot_example(x: np.ndarray, y: np.int64, t: np.ndarray, scale=20) -> None:
    plt.plot(x, t)
    plt.xlim([-scale, scale])
    plt.gca().invert_yaxis()
    plt.xticks([], []), plt.yticks([], [])
    plt.title(f"Label: {y}")


def plot_templates() -> None:
    templates = get_templates()
    x = templates["x"]
    t = templates["t"]
    y = templates["y"]
    plt.gcf().set_size_inches(10, 1)  # Adjust the figure size as needed
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plot_example(x[i], y[i], t, scale=1)
    plt.tight_layout()


def tsne(x: np.ndarray, perplexity: int = 30, seed: int = None):
    return TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(
        x
    )


def tsne_plot(tsne_result, y: np.ndarray) -> None:
    plt.gcf().set_size_inches(8, 6)
    scatter = plt.scatter(
        tsne_result[:, 0], tsne_result[:, 1], c=y, cmap="tab10", alpha=0.7
    )
    plt.colorbar(scatter, label="Class Labels")
    plt.title("t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")


def examples_heatmap(embeddings: list[torch.Tensor]) -> None:
    num_layers = len(embeddings)
    plt.gcf().set_size_inches(10, 2 * num_layers)

    # ===
    # Plot embeddings.
    # ===

    for i, embedding in enumerate(embeddings):
        if len(embedding.shape) == 2:
            # embedding: [N, P] (e.g., [419, 40])
            embedding = embedding.mean(dim=0).reshape(1, -1)  # [1, P]

        elif len(embedding.shape) == 3:
            # embedding: [N, C, P] (e.g., [419, 25, 20])
            embedding = embedding.mean(dim=0)  # [C, P]

        plt.subplot(num_layers, 1, i + 1)
        plt.imshow(
            embedding.cpu().numpy(),
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        plt.colorbar()
        plt.ylabel("Feature")
        plt.xlabel("Part")
        plt.title(f"Layer {i}")

        plt.xticks(
            np.arange(0, embedding.shape[1], step=5 if embedding.shape[1] > 10 else 1)
        )
        plt.yticks(
            np.arange(0, embedding.shape[0], step=5 if embedding.shape[0] > 10 else 1)
        )

    plt.tight_layout()
    plt.show()


# ===
# Training
# ===


def accuracy(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = model(inputs).argmax(-1).cpu().numpy()
    targets_np = targets.cpu().numpy().astype(np.float32)
    return 100 * np.sum(preds == targets_np) / len(targets_np)


def train_val_split(
    dataset: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
    x_train, x_val, y_train, y_val = train_test_split(
        dataset["x"], dataset["y"], test_size=0.2, random_state=0, stratify=dataset["y"]
    )
    x_train = torch.Tensor(x_train).to(torch.float).to(DEVICE)
    x_val = torch.Tensor(x_val).to(torch.float).to(DEVICE)
    y_train = torch.LongTensor(y_train).to(DEVICE)
    y_val = torch.LongTensor(y_val).to(DEVICE)
    return x_train, x_val, y_train, y_val
