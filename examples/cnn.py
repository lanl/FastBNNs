"""Example of training a Bayesian CNN."""

import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from bnn import base, losses, priors, types
from datasets import generic
from models import cnn
from simulation import generators, images, observation


# Create a Bayesian CNN to predict location of a blob in an image.
in_channels = 1
out_channels = 1
hidden_features = 8
kernel_size = 3
stride = 1
padding = "same"
n_hidden_layers = 1
im_size = (8, 8)
nn = torch.nn.Sequential(
    cnn.CNN(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        n_hidden_layers=n_hidden_layers,
        hidden_features=hidden_features,
        activation=torch.nn.LeakyReLU,
    ),
    torch.nn.LeakyReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=np.prod(im_size) * out_channels, out_features=2),
)
bnn = base.BNN(nn=nn)
device = torch.device("cuda")
bnn = bnn.to(device)

# Define a prior (this one applies to all parameters in the model).
prior = priors.Distribution(
    torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
).to(device)

# Define a dataset.
data_generator = generators.Generator(
    simulator=images.gaussian_blobs,
    simulator_kwargs={"im_size": im_size, "sigma": np.array([1.0, 1.0])},
    simulator_kwargs_generator={
        "mu": lambda: np.min(im_size) * (np.random.rand(2)[None, :] - 0.5),
        "amplitude": lambda: np.array([np.random.poisson(lam=100.0)]),
    },
)
noise_tform = observation.NoiseTransform(
    noise_fxn=observation.sensor_noise,
    noise_fxn_kwargs={"sigma": 1.0},
)
n_data = 128 * 10
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
n_epochs = 1000
optimizer = torch.optim.AdamW(bnn.parameters(), lr=1.0e-3)

# Train.
loss_train = []
best_model_state_dict = copy.deepcopy(bnn.state_dict())
best_loss = torch.inf
for epoch in range(n_epochs):
    for batch in dataloader:
        # Forward pass through model.
        optimizer.zero_grad()
        out = bnn(types.MuVar(batch["output"].float().to(device)))

        # Compute loss.
        loss = loss_fn(
            model=bnn,
            input=out[0],
            target=batch["input"]["mu"][:, 0].to(device),
            var=out[1],
        )

        # Update model.
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch+1} of {n_epochs}: loss = {loss}")
    loss_train.append(loss.detach().cpu())
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model_state_dict = copy.deepcopy(bnn.state_dict())

# Plot some examples.
final_model_state_dict = copy.deepcopy(bnn.state_dict())
bnn.load_state_dict(best_model_state_dict)
n_val = 16
dataloader_test = torch.utils.data.DataLoader(dataset=dataset, batch_size=n_val)
bnn = bnn.to("cpu")
data = next(iter(dataloader_test))
output = bnn(types.MuVar(data["output"].float()))

matched_colors = plt.get_cmap("viridis")
fig, ax = plt.subplots()
for n in range(n_val):
    # Plot ground truth.
    ax.plot(
        data["input"]["mu"][n, 0, 1],
        data["input"]["mu"][n, 0, 0],
        ".",
        color=matched_colors(n / n_val),
    )

    # Plot prediction.
    circle = patches.Circle(
        (output[0][n, 1].detach().cpu(), output[0][n, 0].detach().cpu()),
        radius=output[1][n].detach().cpu().mean().sqrt(),
        fill=False,
        color=matched_colors(n / n_val),
    )
    ax.add_patch(circle)

    # Plot a line connection GT to prediction to aid visualization.
    ax.plot(
        [output[0][n, 1].detach().cpu(), data["input"]["mu"][n, 0, 1]],
        [output[0][n, 0].detach().cpu(), data["input"]["mu"][n, 0, 0]],
        color=matched_colors(n / n_val),
    )
ax.plot([], "k.", label="ground truth")
ax.plot([], "ko", markerfacecolor="None", label="predicted +- 1 st. dev.")
ax.set_xlim((-im_size[1] / 2, im_size[1] / 2))
ax.set_ylim((-im_size[0] / 2, im_size[0] / 2))
plt.legend()
plt.show()

print("Done")
