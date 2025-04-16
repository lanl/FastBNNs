"""Example of training a Bayesian MLP."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from bnn import nn_to_bnn, losses
from datasets import generic
from models import mlp
from simulation import generators, polynomials, observation


# Create a Bayesian multilayer perceptron to model a linear function y=mx+b.
in_features = 1
out_features = 1
hidden_features = 1024
n_layers = 1
in_features = 1
out_features = 1
model = mlp.MLP(
    in_features=in_features,
    out_features=out_features,
    n_layers=n_layers,
    hidden_features=hidden_features,
    activation=None,
    # activation=torch.nn.LeakyReLU,
)
nn_to_bnn.convert_to_bnn_(model=model)
device = torch.device("cuda")
model = model.to(device)

# Define a dataset.
data_generator = generators.Generator(
    simulator=polynomials.polynomial,
    simulator_kwargs={"order": 1, "coefficients": np.array([0.0, 1.0])},
    simulator_kwargs_generator={"x": lambda: torch.rand(1) - 0.5},
)
noise_tform = observation.NoiseTransform(
    noise_fxn=observation.add_read_noise,
    noise_fxn_kwargs_generator={
        "sigma": lambda x: 0.1 * (2.0 + torch.cos(2.0 * torch.pi * x))
    },
)
n_data = 1024 * 1
batch_size = 256
dataset = generic.SimulatedData(
    data_generator=data_generator, dataset_length=n_data, transform=noise_tform
)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

# Define optimizer and loss.
# loss_fn = torch.nn.GaussianNLLLoss()  # no KL penalty for now as a demo
loss_fn = losses.ELBO(
    log_likelihood=torch.nn.GaussianNLLLoss(reduction="sum"),
    kl_divergence=losses.KLDivergence(),
    beta=1.0,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

# Train.
n_epochs = 100
loss_train = []
for epoch in range(n_epochs):
    for batch in dataloader:
        # Forward pass through model.
        optimizer.zero_grad()
        out = model(batch["input"]["x"].to(device))

        # Compute loss.
        loss = loss_fn(
            model=model, input=out[0], target=batch["output"].to(device), var=out[1]
        )

        # Update model.
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1} of {n_epochs}: loss = {loss}")
    loss_train.append(loss.detach().cpu())

# Plot some examples.
input = []
output = []
output_gt = []
n_examples = 1000
model = model.to("cpu")
for n in range(n_examples):
    data = dataset[n]
    input.append(data["input"]["x"])
    output_gt.append(data["output"])
input = torch.stack(input, dim=0)
output_gt = torch.cat(output_gt, dim=0)
output = model(input)

fig, ax = plt.subplots()
ax.errorbar(
    input,
    y=output[0].detach().cpu().squeeze(),
    yerr=output[1].detach().cpu().sqrt().squeeze(),
    marker="x",
    linestyle="",
)
ax.plot(input, output_gt, ".", label="GT")
plt.legend()
plt.show()

print("Done")
