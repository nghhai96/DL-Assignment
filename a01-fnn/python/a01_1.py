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

# %% [markdown]
# # Task 1: Implement an MLP
# ## 1a Logistic Regression
#
# The class `LogisticRegression` is located in `a01_functions.py`. You can make
# experimental changes to that class in the other file (`a01_functions.py`). All
# saved changes will be automatically reflected here due to the IPython
# autoreload extension (see below).


# %%
import torch

# %load_ext autoreload
# %autoreload 2
from a01_functions import LogisticRegression, MLP

# %%
# let's test it
logreg = LogisticRegression(3, 2)
x = torch.rand(3)  # input
logreg(x)  # output (log probabilities)
logreg(x).exp()  # output (probabilities)

# %%
# you can access individual parameters as follows
logreg.get_parameter("0_bias")

# %%
# or all of them at once
list(logreg.named_parameters())

# %%
# or directly the tensors stored in the parameters
for par, value in logreg.state_dict().items():
    print(f"{par:<15}= {value}")


# %% [markdown]
# ## 1b MLP
#
# The class `MLP` is also located in `a01_functions.py`. The implementation must
# also be in that file (`a01_functions.py`) as it will be used in later tasks.
#
# Once your implementation is complete, you can proceed with the cells below.

# %%
# here you should see the correct parameter sizes
mlp = MLP([2, 3, 4, 2], torch.relu)
list(mlp.named_parameters())

# %%
# Test your code; we fix the parameters and check the result
with torch.no_grad():
    torch.manual_seed(0)
    for l in range(mlp.num_layers()):
        W, b = mlp.get_parameter(f"{l}_weight"), mlp.get_parameter(f"{l}_bias")
        W[:] = torch.randn(W.shape)
        b[:] = torch.randn(b.shape)

mlp(torch.tensor([-1.0, 2.0]))  # must give: [ 0.8315, -3.6792]

# %%
# You can also evaluate your model on multiple inputs at once. Here "torch.func.vmap"
# produces a function that applies the provided function (mlp#forward) to each row of
# its argument (torch.tensor...).
#
# [[ 0.8315, -3.6792],
# [ 4.8448, -6.8813]]
torch.func.vmap(mlp)(torch.tensor([[-1.0, 2.0], [1.0, -2.0]]))

# %% [markdown]
# ## 1c Batching

# %%
# After you adapted the MLP class, you should get the same results as above.
mlp(torch.tensor([-1.0, 2.0]))  # must give: [ 0.8315, -3.6792]

# %%
# Now without vmap. Only proceed to task 2 once this works correctly.
#
# [[ 0.8315, -3.6792],
# [ 4.8448, -6.8813]]
mlp(torch.tensor([[-1.0, 2.0], [1.0, -2.0]]))

# %%
