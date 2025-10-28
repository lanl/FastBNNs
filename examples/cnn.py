"""Example of training a Bayesian CNN."""

import copy
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.transforms import v2

from fastbnns.models.activations import InverseTransformSampling
from fastbnns.analysis import statistics
from fastbnns.bnn import base, losses, priors, types
from fastbnns.datasets import generic
from fastbnns.simulation import generators, images, observation


# Create a CNN to predict location of a blob in an image.
# Use a custom, data-informed activation to demonstrate FastBNNs support for
# arbitrary nonlinearities.
in_channels = 1
out_channels = 1
hidden_features = 4
kernel_size = 3
stride = 1
padding = "same"
im_size = (8, 8)
x_dist = torch.distributions.Normal(loc=0.0, scale=im_size[0] / 4.0)
nn = torch.nn.Sequential(
    torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=hidden_features,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ),
    torch.nn.ELU(),
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=np.prod(im_size) * hidden_features, out_features=2),
    InverseTransformSampling(
        distribution=x_dist,
        learn_alpha=True,
    ),
)

# Convert `nn` to a BNN, setting learn_var=False for the custom activation
wrapper_kwargs = {"4": {"learn_var": False, "resample_mean": False}}
bnn = base.BNN(nn=nn, convert_in_place=False, wrapper_kwargs=wrapper_kwargs)
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
        "mu": lambda: torch.clamp(
            x_dist.sample(sample_shape=(1, 2)),
            min=-(im_size[0] - 1) / 2,
            max=(im_size[0] - 1) / 2,
        ),
        "amplitude": lambda: np.array([np.random.poisson(lam=100.0)]),
    },
)
noise_tform = observation.NoiseTransform(
    noise_fxn=observation.sensor_noise,
    noise_fxn_kwargs={"sigma": 1.0},
)
scale_tform = v2.Normalize(mean=(8.5,), std=(19.5,))
data_tform = torch.nn.Sequential(
    noise_tform,
    scale_tform,
)
n_data = 128 * 10
batch_size = 128
dataset = generic.SimulatedData(
    data_generator=data_generator,
    dataset_length=n_data,
    transform=data_tform,
    cache=False,
)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
n_data_val = 128 * 10
dataset_val = generic.SimulatedData(
    data_generator=data_generator,
    dataset_length=n_data_val,
    transform=data_tform,
    cache=True,
)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size)

# Define optimizer and loss.
n_batches = n_data // batch_size
loss_fn = losses.ELBO(
    neg_log_likelihood=torch.nn.GaussianNLLLoss(reduction="sum"),
    kl_divergence=losses.KLDivergence(prior=prior),
    beta=1.0 / n_batches,  # see Graves 2011
)
n_epochs = 200
optimizer = torch.optim.AdamW(bnn.parameters(), lr=1.0e-2)

# Train.
loss_train = []
loss_val = []
best_model_state_dict = copy.deepcopy(bnn.state_dict())
best_loss = torch.inf
for epoch in range(n_epochs):
    loss_epoch_train = []
    within_1sigma_train = []
    bnn.train(True)
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

        # Compute gradients and clip to stabilize training.
        loss.backward()

        # Update model.
        optimizer.step()
        loss_epoch_train.append(loss.item())

        # Check predictive variance.
        within_1sigma_train.append(
            statistics.compute_coverage(
                observations=batch["input"]["mu"][:, 0].to(device),
                mu=out[0],
                sigma=out[1].sqrt(),
                alphas=torch.tensor([1.0]),
            ).item()
        )

    # Evaluate on validation set.
    with torch.no_grad():
        bnn.eval()
        loss_epoch_val = []
        within_1sigma_val = []
        for batch in dataloader_val:
            # Forward pass through model.
            out = bnn(types.MuVar(batch["output"].float().to(device)))

            # Compute loss.
            loss = loss_fn(
                model=bnn,
                input=out[0],
                target=batch["input"]["mu"][:, 0].to(device),
                var=out[1],
            )
            loss_epoch_val.append(loss.item())

            # Check predictive variance.
            within_1sigma_val.append(
                statistics.compute_coverage(
                    observations=batch["input"]["mu"][:, 0].to(device),
                    mu=out[0],
                    sigma=out[1].sqrt(),
                    alphas=torch.tensor([1.0]),
                ).item()
            )

    avg_loss_train = np.mean(loss_epoch_train)
    avg_loss_val = np.mean(loss_epoch_val)
    loss_train.append(avg_loss_train)
    loss_val.append(avg_loss_val)
    if avg_loss_val < best_loss:
        best_loss = avg_loss_val
        best_model_state_dict = copy.deepcopy(bnn.state_dict())
    print(
        f"epoch {epoch + 1} of {n_epochs}: train loss = {avg_loss_train:.2f}, val loss = {avg_loss_val:.2f}, {100.0 * np.mean(within_1sigma_val):.2f}% within 1 st. dev."
    )

## Plot some examples.
# Predictions with uncertainty.
final_model_state_dict = copy.deepcopy(bnn.state_dict())
bnn.load_state_dict(best_model_state_dict)
xy_sampler_test = images.GridSamples(n_per_pixel=1, im_size=im_size)
n_data_test = len(xy_sampler_test)
data_generator_test = copy.deepcopy(data_generator)
data_generator_test.simulator_kwargs_generator["mu"] = xy_sampler_test
dataset_test = generic.SimulatedData(
    data_generator=data_generator_test,
    dataset_length=n_data_test,
    transform=data_tform,
    cache=True,
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test, batch_size=batch_size
)
data = next(iter(dataloader_test))
with torch.no_grad():
    bnn.eval()
    output = bnn(types.MuVar(data["output"].float().to(device)))

matched_colors = plt.get_cmap("viridis")
matched_colors = [matched_colors(n / n_data_test) for n in range(n_data_test)]
random.shuffle(matched_colors)  # shuffle to move colors apart in plot
fig, ax = plt.subplots()
for n in range(n_data_test):
    # Plot ground truth.
    ax.plot(
        data["input"]["mu"][n, 0, 1],
        data["input"]["mu"][n, 0, 0],
        ".",
        color=matched_colors[n],
    )

    # Plot prediction.
    circle = patches.Circle(
        (output[0][n, 1].detach().cpu(), output[0][n, 0].detach().cpu()),
        radius=output[1][n].detach().cpu().mean().sqrt(),
        fill=False,
        color=matched_colors[n],
    )
    ax.add_patch(circle)

    # Plot a line connection GT to prediction to aid visualization.
    ax.plot(
        [output[0][n, 1].detach().cpu(), data["input"]["mu"][n, 0, 1]],
        [output[0][n, 0].detach().cpu(), data["input"]["mu"][n, 0, 0]],
        color=matched_colors[n],
    )
ax.plot([], "k.", label="ground truth")
ax.plot([], "ko", markerfacecolor="None", label="predicted +- 1 st. dev.")
ax.set_xlim((-im_size[1] / 2, im_size[1] / 2))
ax.set_ylim((-im_size[0] / 2, im_size[0] / 2))
plt.legend()
plt.show()
print("done")
