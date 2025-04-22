"""Example of training a Bayesian MLP in Lightning."""

import copy

import lightning as L
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
hidden_features = 32
n_hidden_layers = 1
in_features = 1
out_features = 1
nn = mlp.MLP(
    in_features=in_features,
    out_features=out_features,
    n_hidden_layers=n_hidden_layers,
    hidden_features=hidden_features,
    activation=torch.nn.LeakyReLU,
)
bnn = base.BNN(nn=nn)

# Define a prior (this one applies to all parameters in the model).
prior = priors.Distribution(
    torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
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
        "sigma": lambda x: 0.1 + 0.2 * (torch.cos(2.0 * torch.pi * x) ** 2)
    },
)
n_data = 1024 * 10
batch_size = 128
dataset = generic.SimulatedData(
    data_generator=data_generator, dataset_length=n_data, transform=noise_tform
)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True
)

# Define optimizer and loss.
n_batches = n_data // batch_size
loss_fn = losses.ELBO(
    neg_log_likelihood=torch.nn.GaussianNLLLoss(reduction="sum"),
    kl_divergence=losses.KLDivergence(prior=prior),
    beta=1.0 / n_batches,  # see Graves 2011
)

# Prepare a Lightning module.
bnn_lightning = base.BNNLightning(bnn=bnn, loss=loss_fn)

# Train.
n_epochs = 100
trainer = L.Trainer(max_epochs=n_epochs, check_val_every_n_epoch=n_epochs)
trainer.fit(
    model=bnn_lightning, train_dataloaders=dataloader, val_dataloaders=dataloader
)

# Plot some examples.
input = []
output = []
n_examples = 1000
bnn = bnn.to("cpu")
dataset.data_generator.simulator_kwargs_generator["x"] = lambda: 2.0 * (
    torch.rand(1) - 0.5
)
for n in range(n_examples):
    data = dataset[n]
    input.append(data["input"]["x"])
input = torch.stack(input, dim=0)
output = bnn(input)

x, sort_inds = torch.sort(input.cpu().squeeze())
y = output[0].detach().cpu().squeeze()[sort_inds]
yerr = output[1].detach().cpu().sqrt().squeeze()[sort_inds]
y_gt = data_generator.simulator(x=x, **data_generator.simulator_kwargs)
yerr_gt = noise_tform.noise_fxn_kwargs_generator["sigma"](x)
fig, ax = plt.subplots()
ax.plot(x, y_gt, color="k", linestyle=":", label="ground truth")
ax.fill_between(
    x=x,
    y1=y_gt - yerr_gt,
    y2=y_gt + yerr_gt,
    alpha=0.5,
    color="k",
    hatch="x",
    label="true uncertainty",
)
ax.fill_between(
    x=x,
    y1=y - yerr,
    y2=y + yerr,
    alpha=0.5,
    color="m",
    label="pred. uncertainty",
)
ax.plot(x, y, marker=".", linestyle="", label="predicted mean")
ax.set_ylim((y_gt.min(), y_gt.max()))
plt.legend()
plt.show()

print("Done")
