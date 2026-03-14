# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# import helper functions
import sys, os

sys.path.append(os.getcwd())
# %load_ext autoreload
# %autoreload 2
from a01_helper import *
from a01_functions import train1, MLP  # MLP is implicitly required for `torch.load`

# %% [markdown]
# # 2 Multi-Layer Feed-Forward Neural Networks
# ## 2a Conjecture how an FNN fit will look like

# %%
# here is the one-dimensional dataset that we will use
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plt.legend(loc="upper right")

# Use this function to save your plot.
saveplot("a01_2_data.pdf")

# %% [markdown]
# ## 2b Train with 2 hidden units

# %%
# Let's fit the model with one hidden layer consisting of 2 units.
model = train1([2], nreps=1)
print("Training error:", F.mse_loss(y1, model(X1)).item())
print("Test error    :", F.mse_loss(y1test, model(X1test)).item())

# %%
# plot the data and the fit
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(torch.linspace(0, 13, 500).unsqueeze(1), model)

saveplot("a01_2_fit-2-neurons.pdf")

# %%
# The weight matrices and bias vectors can be read out as follows. If you want, use
# these parameters to compute the output of the network (on X1) directly and compare to
# vmap(model)(X1).
for par, value in model.state_dict().items():
    print(f"{par:<15}= {value}")

# %%
# now repeat this multiple times
# TODO: YOUR CODE HERE

# %%
# From now on, always train multiple times (nreps=10 by default) and report best model.
model = train1([2], nreps=10)
print("Training error:", F.mse_loss(y1, model(X1)).item())
print("Test error    :", F.mse_loss(y1test, model(X1test)).item())

# %% [markdown]
# ## 2c Width

# %%
# Experiment with different hidden layer sizes. To avoid recomputing
# models, you may want to save your models using torch.save(model, filename) and
# load them again using torch.load(filename).
# TODO: YOUR CODE HERE

# %% [markdown]
# ## 2d Distributed representations

# %%
# train a model to analyze
model = train1([2])

# TODO: YOUR CODE HERE

# %%
# plot the fit as well as the outputs of each neuron in the hidden
# layer (scale for the latter is shown on right y-axis)
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(torch.linspace(0, 13, 500).unsqueeze(1), model, hidden=True, scale=False)
saveplot("a01_2_distributed-reps.pdf")

# %%
# plot the fit as well as the outputs of each neuron in the hidden layer, scaled
# by its weight for the output neuron (scale for the latter is shown on right
# y-axis)

nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(torch.linspace(0, 13, 500).unsqueeze(1), model, hidden=True, scale=True)
plt.legend(loc="upper right")
plt.tight_layout()
saveplot("a01_2_distributed-reps-scaled.pdf")

# %% [markdown]
# ## 2e Experiment with different optimizers (optional)

# %%
# PyTorch provides many gradient-based optimizers; see
# https://pytorch.org/docs/stable/optim.html. You can use a PyTorch optimizer
# as follows.
train_adam = lambda model, **kwargs: fnn_train(
    X1, y1, model, optimizer=torch.optim.Adam(model.parameters(), lr=0.01), **kwargs
)
model = train1([50], nreps=1, train=train_adam, max_epochs=5000, tol=1e-8, verbose=True)

# %%
# Experiment with different number of layers and activation functions. Here is
# an example with three hidden layers (of sizes 4, 5, and 6) and ReLU activations.
#
# You can also plot the outputs of the hidden neurons in the first layer (using
# the same code above).
model = train1([4, 5, 6], nreps=50, phi=F.relu)
nextplot()
plot1(X1, y1, label="train")
plot1(X1test, y1test, label="test")
plot1fit(torch.linspace(0, 13, 500).unsqueeze(1), model)
print("Training error:", F.mse_loss(y1, model(X1)).item())
print("Test error    :", F.mse_loss(y1test, model(X1test)).item())
