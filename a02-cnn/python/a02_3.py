# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import numpy as np
import torch

from a02_functions import SimpleCNN, train_model
from a02_helper import get_raw_data, tsne, tsne_plot, DEVICE, examples_heatmap, nextplot

# %load_ext autoreload
# %autoreload 2


# %% [markdown]
# # Task 3: Visualization

# %% [markdown]
# ### t-SNE for input data

# %%
perplexity = 30
n_examples = 1000  # use lower number for faster computation

np.random.seed(0)
rand_indices = np.random.choice(4000, min(4000, n_examples))

data = get_raw_data()
x = data["x"][rand_indices]
y = data["y"][rand_indices]
x_test = data["x_test"]
y_test = data["y_test"]

# %%
x_tsne = tsne(x, perplexity=perplexity)

# %%
nextplot()
tsne_plot(x_tsne, y)

# %% [markdown]
# ### t-SNE for embeddings of training data

# %%
torch.manual_seed(0)
cnn = SimpleCNN()
_ = train_model(data, cnn)


# %%
# Before you start, populate `embedding` during the forward pass (in SimpleCNN). It
# should be a list with one element per layer, each being a tensor of form examples x
# embeddings.
cnn.store_embeddings = True
with torch.no_grad():
    cnn(torch.Tensor(x).to(DEVICE))
train_embeddings = [e.cpu().numpy() for e in cnn.embeddings]

# TODO: your code here


# %%

# %% [markdown]
# ### t-SNE for embeddings of test data

# %%
# TODO: your code here


# %% [markdown]
# ### Embeddings Heatmap

# %% [markdown]
# Following the example below, create a plot that visualizes the average activation
# strength over all examples from a certain class `cls` for all features (channels).
#
# It might help to take a look into the plot function `embedding_heatmap`. Let `cls = 0`
# for instance, i.e., let us start with the "digit" 0. The dataset (depending on the
# seed and generation parameters) contains 419 examples for that digit (approximately
# 10% of all training examples). Since the length of each "digit" is 40, i.e., 40
# *parts*, the input dimensionality is $(419, 1, 40)$. Each example is one dimensional,
# i.e., we only have single feature per example. In the first heatmap (*Input*), we
# average over all examples and hence see the average value of the only feature (y-axis)
# for the individual parts (for each "time step"; x-axis).
#
# After first layer in your CNN (when `stride = 2` and `channels = 25`), the
# dimensionality of its output is $(419, 25, 20)$, so the number of parts has halved due
# to the stride, but there are now 25 features due to the number of channels.
#
# Now, in the heatmap below, we average over all examples of class `cls`. This means
# that the heatmap shows the average activations over all examples of that class `cls`
# for all features (y-axis) and parts (x-axis).

# %%
cnn.store_embeddings = True
with torch.no_grad():
    cnn(torch.Tensor(x).to(DEVICE))

train_embeddings = (
    cnn.embeddings
)  # a list of per-layer embeddings, each of form example x embedding

cls = 0  # investigate different classes by changing this index
cls_examples = torch.Tensor(x[y == cls])
cls_train_embeddings = [e[y == cls] for e in train_embeddings]

nextplot()
examples_heatmap([cls_examples] + cls_train_embeddings[:-1])

# TODO: your code here
