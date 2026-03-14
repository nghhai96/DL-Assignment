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
import math
import torch
from torch import nn
import torch.nn.functional as F

# %% [markdown]
# # Task 1: Implement an MLP
# ## 1a Logistic Regression


# %%
# `nn.Module` is the superclass of all PyTorch models.
class LogisticRegression(nn.Module):
    """A logistic regression model.

    Parameters
    ----------
    D number of inputs
    C number of classes
    """

    # the definition of all parameters the model uses happens here, i.e., during
    # initialization
    def __init__(self, D, C):
        super().__init__()

        # Create and initialize model parameters. For (multinomial) logistic regression,
        # we have a DxC-dimensional weight matrix W and a C-dimensional bias b.
        self.W = torch.randn(D, C) / math.sqrt(D)
        self.b = torch.randn(C) / math.sqrt(C)

        # Model parameters must be registered to PyTorch as follows. Here we provide
        # a useful name that helps to access/analyze the model later on.
        self.register_parameter("0_weight", nn.Parameter(self.W))
        self.register_parameter("0_bias", nn.Parameter(self.b))

    # the forward function computes the model output for the provided (for this
    # assignment: single) input
    def forward(self, x):
        eta = self.W.t() @ x + self.b
        logprob = F.log_softmax(eta, dim=-1)
        return logprob


# %% [markdown]
# ## 1b MLP


# %%
class MLP(nn.Module):
    """A fully-connected MLP.

    Parameters
    ----------

    sizes Contains the layer sizes. The first entry is the number of inputs, the last
    entry the number of outputs. All entries in between correspond to the number of
    units in the respective hidden layer. E.g., [2,5,7,1] means: 2 inputs -> 5D hidden
    layer -> 7D hidden layer -> 1 output.

    phi Activation function used in every hidden layer (the output layer is linear).

    """

    def __init__(self, sizes: list[int], phi=F.sigmoid):
        super().__init__()

        # let's remember the specification in this model
        self.sizes = sizes
        self.phi = phi

        # Initialize and register the parameters. Follow the naming scheme used for
        # logistic regression above, i.e., the layer-i weights should be named "i_weight" and
        # "i_bias".
        #
        # TODO: YOUR CODE HERE
        for i in range(self.num_layers()):
            self.W = torch.randn(self.sizes[i], self.sizes[i+1]) / math.sqrt(self.sizes[i])
            self.b = torch.randn(self.sizes[i+1]) / math.sqrt(self.sizes[i+1])
            self.register_parameter(f"{i}_weight", nn.Parameter(self.W))
            self.register_parameter(f"{i}_bias", nn.Parameter(self.b))

    def num_layers(self):
        """Number of layers (excluding input layer)"""
        return len(self.sizes) - 1

    def forward(self, x):
        # TODO: YOUR CODE HERE
        eta = x
        if x.shape[0] == 1: # If input size 1 x D then eta = W.t() @ eta + b
            for i in range(self.num_layers()):
                W = self.get_parameter(f"{i}_weight")
                b = self.get_parameter(f"{i}_bias")
                eta = W.t() @ eta + b
                if i < self.num_layers() - 1: #Not apply Activation Function at the last layer
                    eta = self.phi(eta)
        else: # If input size N x D then eta = eta @ W + b
            for i in range(self.num_layers()):
                W = self.get_parameter(f"{i}_weight")
                b = self.get_parameter(f"{i}_bias")
                eta = eta @ W + b
                if i < self.num_layers() - 1:  # Not apply Activation Function at the last layer
                    eta = self.phi(eta)
        return eta


# %% [markdown]
# # 2 Multi-Layer Feed-Forward Neural Networks
# ## 2b Train with 2 hidden units

# %%
# Training code. You do not need to modify this code.
from a01_helper import train_scipy, X1, y1

train_bfgs = lambda model, **kwargs: train_scipy(X1, y1, model, **kwargs)


def train1(hidden_sizes, nreps=10, phi=F.sigmoid, train=train_bfgs, **kwargs):
    """Train an FNN.

    hidden_sizes is a (possibly empty) list containing the sizes of the hidden layer(s).
    nreps refers to the number of repetitions.

    """
    best_model = None
    best_cost = math.inf
    for rep in range(nreps):
        model = MLP([1] + hidden_sizes + [1], phi)  # that's your model!
        print(f"X1 shape: {X1.shape}")
        print(f"Repetition {rep: 2d}: ", end="")
        model = train(model, **kwargs)
        mse = F.mse_loss(y1, model(X1)).item()
        if mse < best_cost:
            best_model = model
            best_cost = mse
        print(f"best_cost={best_cost:.3f}")

    return best_model
