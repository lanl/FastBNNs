"""Example of training a Bayesian MLP."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

from bnn import base, losses, priors
from datasets import generic
from models import mlp
from simulation import generators, polynomials, observation


# Create a Bayesian multilayer perceptron to model a linear function y=mx+b.
in_features = 1
out_features = 1
hidden_features = 128
n_hidden_layers = 2
in_features = 1
out_features = 1
nn = mlp.MLP(
    in_features=in_features,
    out_features=out_features,
    n_hidden_layers=n_hidden_layers,
    hidden_features=hidden_features,
    activation=torch.nn.LeakyReLU,
    # activation=None,
)
bnn = base.BNN(nn=nn)
device = torch.device("cuda")
bnn = bnn.to(device)

# Define a prior (this one applies to all parameters in the model).
prior = priors.SpikeSlab(
    loc=torch.tensor([0.0, 0.0], device=device),
    scale=torch.tensor([0.01, 5.0], device=device),
    probs=torch.tensor([0.5, 0.5], device=device),
)

# Define a dataset.
data_generator = generators.Generator(
    simulator=polynomials.polynomial,
    simulator_kwargs={"order": 1, "coefficients": np.array([0.0, 1.0])},
    simulator_kwargs_generator={"x": lambda: torch.rand(1) - 0.5},
)
noise_tform = observation.NoiseTransform(
    noise_fxn=observation.add_read_noise,
    noise_fxn_kwargs_generator={
        "sigma": lambda x: 0.1
        + 0.2 * (torch.cos(2.0 * torch.pi * x) ** 2)
        # "sigma": lambda x: 0.1
    },
)
n_data = 1024 * 10
batch_size = 128
dataset = generic.SimulatedData(
    data_generator=data_generator, dataset_length=n_data, transform=noise_tform
)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

# Define optimizer and loss.
n_batches = n_data // batch_size
loss_fn = losses.ELBO(
    neg_log_likelihood=torch.nn.GaussianNLLLoss(reduction="sum"),
    kl_divergence=losses.KLDivergence(prior=prior),
    beta=1.0 / n_batches,  # see Graves 2011
)
n_epochs = 100
optimizer = torch.optim.AdamW(bnn.parameters(), lr=1.0e-2)

# Train.
loss_train = []
best_model_state_dict = copy.deepcopy(bnn.state_dict())
best_loss = torch.inf
for epoch in range(n_epochs):
    for batch in dataloader:
        # Forward pass through model.
        optimizer.zero_grad()
        out = bnn(batch["input"]["x"].to(device))

        # Compute loss.
        loss = loss_fn(
            model=bnn, input=out[0], target=batch["output"].to(device), var=out[1]
        )

        # Update model.
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1} of {n_epochs}: loss = {loss}")
    loss_train.append(loss.detach().cpu())
    if loss.item() < best_loss:
        best_model_state_dict = copy.deepcopy(bnn.state_dict())

# Plot some examples.
bnn.load_state_dict(best_model_state_dict)
input = []
output = []
output_gt = []
n_examples = 1000
bnn = bnn.to("cpu")
dataset.data_generator.simulator_kwargs_generator["x"] = lambda: 2.0 * (
    torch.rand(1) - 0.5
)
for n in range(n_examples):
    data = dataset[n]
    input.append(data["input"]["x"])
    output_gt.append(data["output"])
input = torch.stack(input, dim=0)
output_gt = torch.cat(output_gt, dim=0)
output = bnn(input)

fig, ax = plt.subplots()
ax.errorbar(
    input,
    y=output[0].detach().cpu().squeeze(),
    yerr=output[1].detach().cpu().sqrt().squeeze(),
    marker="x",
    linestyle="",
)
ax.plot(input, output_gt, ".", label="GT")
ax.set_ylim((output_gt.min(), output_gt.max()))
plt.legend()
plt.show()

print("Done")
