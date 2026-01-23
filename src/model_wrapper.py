# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import Optional

import numpy as np
import plotly.express as px
import torch
import yaml
from dwave.plugins.torch.models import (
    DiscreteVariationalAutoencoder,
    GraphRestrictedBoltzmannMachine,
)
from einops import rearrange
from plotly import graph_objects as go
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from .decoder import CNNDecoder as Decoder
from .encoder import CNNEncoder
from .encoder import EncoderWithNHF as Encoder
from .utils.common import get_coeff, get_exp_coeff, get_sampler_and_sampler_kwargs


def get_dataset(batch_size: int, dataset_size: Optional[int] = None) -> DataLoader:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: (x > 0.5).float())]
    )

    # For training data with augmentation
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    if dataset_size:
        mnist_dataset = torch.utils.data.random_split(
            mnist_dataset, [dataset_size, len(mnist_dataset) - dataset_size]
        )[0]
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


def display_dataset(dataset: DataLoader, num_rows: int) -> go.Figure:
    batch = next(iter(dataset))[0]
    reconstruction_tensor_for_plot = make_grid(batch.cpu(), nrow=num_rows)
    fig = px.imshow(reconstruction_tensor_for_plot.permute(1, 2, 0))

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        margin={"t": 0, "l": 0, "b": 0, "r": 0},
        paper_bgcolor="black",
        plot_bgcolor="black",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


class LatentToDiscrete:
    def __init__(self):
        self.flip_probability = 0.0  # This is changed during training
        self.noise_level = 0.0  # This is changed during training

    def __call__(self, latents: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        assert n_samples == 1, "Only n_samples=1 is supported in this LatentToDiscrete"
        # print("In latent_to_discrete:")
        # print("Latents shape:", latents.shape)
        discretes = torch.sign(latents)
        # Return the straight-through discrete variables
        spins = (latents + (discretes - latents).detach()).unsqueeze(1)
        # Randomly flip some of the spins with probability self.flip_probability
        if self.flip_probability > 0.0:
            flip_mask = torch.rand_like(spins) < self.flip_probability
            spins = torch.where(flip_mask, -spins, spins)
        if self.noise_level > 0.0:
            noise = torch.randn_like(spins) * self.noise_level
            spins += noise
        return spins


class TrainingError(Exception):
    """Error when training the model."""


class ModelWrapper:
    """Container class for the discrete VAE w. GRBM model.

    Args:
        n_latents: The number of latent variables in the model.
    """

    def __init__(
        self,
        qpu: str,
        n_latents_per_pixel: int,
        training_parameter_file: str = None,
    ) -> None:
        self.qpu: str = qpu
        self.n_latents_per_pixel: int = n_latents_per_pixel
        self._shapex: tuple[int, int, int] = (1, 28, 28)
        self._shapez: tuple[int, int, int] = (self.n_latents_per_pixel, 3, 3)
        self._factors: list[int] = [4, 4, 4]

        self._dvae = None
        self._grbm = None

        self._device = None
        self.sampler = None
        self.sampler_kwargs = None

        self._dvae_optimizer = None
        self._grbm_optimizer = None

        self._dataloader = None

        self.losses = {"mse_losses": [], "dvae_losses": []}

        if not training_parameter_file:
            training_parameter_file = "src/training_parameters.yaml"

        with open(training_parameter_file, "r") as f:
            self._params = yaml.safe_load(f)

    def __getattr__(self, name: str):
        if name in self._params:
            return self._params[name]
        return super().__getattribute__(name)

    def save(self, file_path: Optional[str] = None) -> None:
        """Save the model and configs.

        Args:
            file_path: Relative path to the folder where the model should be saved.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        file_path.mkdir(exist_ok=True)

        # Save the model
        torch.save(self._dvae.state_dict(), file_path / "dvae.pth")
        # Save the RBM
        torch.save(self._grbm.state_dict(), file_path / "grbm.pth")

    def load(self, file_path: str) -> None:
        """Load and reconstruct autoencoder from saved models and configs.

        Args:
            file_path: Relative path to the folder containing the saved model.
        """
        self.setup()
        self._load_dataset(batch_size=self.BATCH_SIZE, dataset_size=self.DATASET_SIZE)

        # currently assuming config and and model have same base name
        dvae = torch.load(file_path / "dvae.pth", weights_only=False, map_location=self._device)
        self._dvae.load_state_dict(dvae.state_dict())
        self._grbm = torch.load(
            file_path / "grbm.pth", weights_only=False, map_location=self._device
        )

    def setup(self) -> None:
        """Initial setup for the VAE and GRBM."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dvae = DiscreteVariationalAutoencoder(
            encoder=Encoder(
                main_encoder=CNNEncoder(shapex=self._shapex, factors=self._factors),
                n_latents_per_pixel=self.n_latents_per_pixel,
                use_orthogonal_layers=False,
                num_orthogonal_layers=1,
                use_bias_in_orthogonal_layers=True,
                flatten_all_features=False,
                disable_hashmap=True,
            ),
            decoder=Decoder(
                shapez=self._shapez,
                factors=self._factors + [4],
                shapex=self._shapex,
                codebook_dim=64 * 4,
            ),
            latent_to_discrete=LatentToDiscrete(),
        )

        self._dvae = dvae.to(self._device)
        self.sampler, graph, self.linear_range, self.quadratic_range, self.sampler_kwargs = (
            get_sampler_and_sampler_kwargs(
                num_reads=self.NUM_READS,
                annealing_time=self.ANNEALING_TIME,
                num_sweeps=self.NUM_SWEEPS,
                n_latents=self.n_latents_per_pixel * self._shapez[1] * self._shapez[2],
                random_seed=self.RANDOM_SEED,
                use_qpu=True,
                solver_name=self.qpu,
                parallel_embedding=False,
            )
        )

        grbm = GraphRestrictedBoltzmannMachine(
            graph.nodes(),
            graph.edges(),
        )
        self._grbm = grbm.to(self._device)
        self._grbm_sampleset = None

        self._dvae_optimizer = torch.optim.Adam(
            self._dvae.parameters(),
            lr=self.AUTOENCODER_INITIAL_LR,
        )
        self._grbm_optimizer = torch.optim.Adam(
            self._grbm.parameters(),
            lr=self.BM_INITIAL_LR,
        )

    def _load_dataset(self, batch_size: int, dataset_size: Optional[int] = None) -> None:
        """Load the MNIST dataset and create the dataloader.

        Args:
            batch_size: The batch size to use.
            dataset_size: The number of images to use for training.
                Default (``None``) uses all available images.
        """
        self._dataloader = get_dataset(batch_size, dataset_size)

    def train_init(
        self,
    ) -> None:
        """Initialize the model for training."""
        self.losses["mse_losses"].clear()
        self.losses["dvae_losses"].clear()

        # set the random seed for reproducibility
        torch.manual_seed(self.RANDOM_SEED)

        if self._dvae is None or self._grbm is None:
            self.setup()

        if self._dataloader is None:
            self._load_dataset(batch_size=self.BATCH_SIZE, dataset_size=self.DATASET_SIZE)

    def step(self, batch: tuple[torch.Tensor, torch.Tensor], opt_step: int) -> torch.Tensor:
        """Train the model on a single batch.

        Args:
            batch: The batch to train on.
            opt_step: The current optimization step.

        Returns:
            torch.Tensor: MSE loss from training step.
        """
        if not self._tpar.get("init_done", True):
            raise TrainingError("Initialization required before training.")

        images, _ = batch
        images = images.to(self._device)
        self._dvae.train()
        self._grbm.train()
        grbm_coeff = get_exp_coeff(
            opt_step, n_rampup=10_000, min_coeff=1e-6, max_coeff=0.1, delay=1_000, cutoff=20_000
        )
        cov_matching_coeff = get_coeff(opt_step, n_rampup=5_000, min_coeff=1e-2, max_coeff=10)
        bimodal_coeff = get_coeff(opt_step, n_rampup=8_000, min_coeff=1e-3, max_coeff=0.1)
        l1_coeff = get_coeff(opt_step, n_rampup=10_000, min_coeff=1e-6, max_coeff=1.0)
        flip_prob = get_coeff(opt_step, n_rampup=1_000, min_coeff=0.0, max_coeff=0.1)
        noise_level = get_coeff(opt_step, n_rampup=1_000, min_coeff=1e-4, max_coeff=0.2)
        self._dvae.latent_to_discrete.flip_probability = flip_prob
        self._dvae.latent_to_discrete.noise_level = noise_level

        latents, spins, reconstructed_images = self._dvae(images, n_samples=1)
        reconstructed_images = reconstructed_images.squeeze(1)
        spins = spins.squeeze(1)
        if flip_prob > 0.0 or noise_level > 0.0:
            # The flipped or noised spins go into the decoder to avoid overfitting, but the
            # original clean spins should be used for all other computations:
            spins = torch.sign(latents)
            spins = latents + (spins - latents).detach()

        self._dvae_optimizer.zero_grad()

        l1_loss = torch.nn.functional.l1_loss(reconstructed_images, images)
        l2_loss = torch.nn.functional.mse_loss(reconstructed_images, images)
        bimodal_loss = torch.nn.functional.mse_loss(latents, spins)

        if (
            grbm_coeff > 0.0
            or self._grbm_sampleset is None
            or (grbm_coeff == 0.0 and opt_step % 10 == 0)
        ):
            self._grbm_sampleset = self._grbm.sample(
                self.sampler,
                as_tensor=False,
                prefactor=0.16,
                sample_params=self.sampler_kwargs,
                device=self._device,
                linear_range=self.linear_range,
                quadratic_range=self.quadratic_range,
            )
        grbm_samples = self._grbm.sampleset_to_tensor(self._grbm_sampleset, device=self._device)
        cov_latents = torch.cov(latents.flatten(start_dim=1).T)
        cov_grbm = torch.cov(grbm_samples.detach().T)
        cov_matching_loss = torch.nn.functional.mse_loss(cov_latents, cov_grbm)

        mean_latents = torch.mean(latents.flatten(start_dim=1), dim=0)
        mean_grbm = torch.mean(grbm_samples.detach(), dim=0)
        cov_matching_loss += torch.nn.functional.mse_loss(mean_latents, mean_grbm)

        loss = (
            l2_loss
            + l1_coeff * l1_loss
            + bimodal_coeff * bimodal_loss
            + cov_matching_coeff * cov_matching_loss
        )
        loss.backward()
        self._dvae_optimizer.step()

        if grbm_coeff > 0.0:
            self._grbm_optimizer.zero_grad()
            grbm_loss = self._grbm.quasi_objective(
                s_observed=spins.detach().flatten(start_dim=1), s_model=grbm_samples
            )
            scaled_grbm_loss = grbm_coeff * grbm_loss
            scaled_grbm_loss.backward()
            self._grbm_optimizer.step()
        self.losses["mse_losses"].append(l2_loss.item())
        self.losses["dvae_losses"].append(loss.item())
        return loss

    def generate_output(self, sharpen: bool = False, save_to_file: str = "") -> go.Figure:
        """Generate output images from trained model.

        Args:
            Whether to sharpen the output images by binarization.

        Returns:
            go.Figure: Plotly figure.
        """
        images_per_row = 16
        print("Generating output images...")
        self._dvae.eval()
        self._grbm.eval()
        print("DVAE and GRB in eval mode")

        with torch.no_grad():
            grbm_samples = self._grbm.sample(
                self.sampler,
                as_tensor=True,
                prefactor=0.16,
                device=self._device,
                linear_range=self.linear_range,
                quadratic_range=self.quadratic_range,
                sample_params=self.sampler_kwargs,
            )
            print("Got samples from GRBM")
            latents_from_grbm = grbm_samples.view(
                grbm_samples.size(0), 1, self.n_latents_per_pixel, self._shapez[1], self._shapez[2]
            )
            print("Reshaped latents from GRBM with shape", latents_from_grbm.shape)
            print(self._dvae.decoder)
            images = self._dvae.decoder(latents_from_grbm).squeeze(1).clip(0.0, 1.0).detach().cpu()
            print("Decoded images from latents")
        if sharpen:
            images = (images > 0.5).float()

        generation_tensor_for_plot = make_grid(images, nrow=images_per_row)
        print("Got grid for plotting")

        fig = px.imshow(generation_tensor_for_plot.permute(1, 2, 0))

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})

        if save_to_file:
            with open(save_to_file, "w") as f:
                f.write(fig.to_json())

        return fig

    def generate_loss_plot(
        self,
        save_to_file_mse: str = "",
        save_to_file_total: str = "",
        old_loss_data: Optional[list] = None,
    ) -> tuple[go.Figure, go.Figure]:
        """Generate the loss plots for MSE and DVAE loss.

        Returns:
            go.Figure: The Mean Squared Error losses plot.
            go.Figure: The total losses plot.
        """
        if old_loss_data:
            mse_losses = old_loss_data["mse_losses"] + self.losses["mse_losses"]
            dvae_losses = old_loss_data["dvae_losses"] + self.losses["dvae_losses"]
        else:
            mse_losses = self.losses["mse_losses"]
            dvae_losses = self.losses["dvae_losses"]

        fig_mse = go.Figure()
        fig_total = go.Figure()

        fig_mse.add_trace(go.Scatter(x=list(range(len(mse_losses))), y=mse_losses))
        fig_total.add_trace(go.Scatter(x=list(range(len(mse_losses))), y=dvae_losses))

        # Update xaxis properties
        fig_mse.update_xaxes(title_text="Batch")
        fig_mse.update_yaxes(title_text="Loss")

        # Update yaxis properties
        fig_total.update_xaxes(title_text="Batch")
        fig_total.update_yaxes(title_text="Loss")

        fig_mse.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})
        fig_total.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})

        if save_to_file_mse:
            with open(save_to_file_mse, "w") as f:
                f.write(fig_mse.to_json())

        if save_to_file_total:
            with open(save_to_file_total, "w") as f:
                f.write(fig_total.to_json())

        return fig_mse, fig_total

    def generate_reconstucted_samples(
        self, sharpen: bool = False, save_to_file: str = ""
    ) -> go.Figure:
        """Generate reconstructed images from training data.

        Args:
            Whether to sharpen the output images by binarization.

        Returns:
            go.Figure: A figure showing the comparison between original and reconstructed digits.
        """
        images_per_row = 16
        # Now we use the trained autoencoder both to generate new samples as well as to
        # show the reconstruction of the input samples.
        batch = next(iter(self._dataloader))[0]
        self._dvae.eval()

        _, _, reconstructed_batch = self._dvae(batch.to(self._device))
        # reconstructed_batch[:, :, :, :, -1] = 1.0 # what is this?
        images = make_grid(
            rearrange(
                [batch.cpu(), reconstructed_batch.clip(0.0, 1.0).squeeze(1).cpu()],
                "i b c h w -> (b i) c h w",
            ),
            nrow=images_per_row,
            padding=0,
        )

        if sharpen:
            images = (images > 0.5).float()

        fig = px.imshow(images.permute(1, 2, 0))

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})

        if save_to_file:
            with open(save_to_file, "w") as f:
                f.write(fig.to_json())

        return fig
