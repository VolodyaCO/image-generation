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
from abc import ABC, abstractmethod
from math import prod

import torch
import torch.nn as nn
from einops import rearrange


class PureCNNEncoder(nn.Module):
    def __init__(self, shapex: tuple[int, int, int], factors: list[int]):
        super().__init__()
        self.shapex = shapex
        self.factors = factors
        cx, hx, wx = shapex
        assert hx == wx, "Height and width of the images must be equal"
        out_factor = 1
        layers = []
        hi, wi = hx, wx
        for nl, factor in enumerate(factors):
            cin = cx * out_factor
            out_factor *= factor
            cout = cx * out_factor
            layers.append(nn.LayerNorm([cin, hi, wi]))
            hi = hi // 2
            wi = wi // 2
            # Increase number of channels, keep image dimension:
            layers.append(nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1))
            # Keep number of channels, decrease image dimension:
            layers.append(nn.Conv2d(cout, cout, kernel_size=4, stride=2, padding=1))
            if nl < len(factors) - 1:
                layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(cout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(ABC, nn.Module):
    @property
    @abstractmethod
    def cout(self) -> int: ...

    @property
    @abstractmethod
    def n_latent_dims(self) -> int: ...


class CNNEncoder(Encoder):
    def __init__(self, shapex: tuple[int, int, int], factors: list[int]):
        super().__init__()
        self.conv = PureCNNEncoder(shapex, factors)
        self._cout = prod(factors, start=shapex[0])
        hz = wz = shapex[1] // (2 ** len(factors))
        self._n_latent_dims = self._cout * hz * wz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    @property
    def cout(self) -> int:
        return self._cout

    @property
    def n_latent_dims(self) -> int:
        return self._n_latent_dims


class DeepHashMap(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 1,
        use_orthogonal_layers: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        # We interpolate between input_dim and output_dim over n_layers:
        dims = torch.linspace(input_dim, output_dim, n_layers + 1).int().tolist()
        layers = []
        if use_orthogonal_layers:
            # TODO: once https://github.com/dwavesystems/dwave-pytorch-plugin/pull/57 is merged,
            # we can import GivensOrthogonalLinear from there
            raise NotImplementedError("GivensOrthogonalLinear is not yet available.")
            # layer_constructor = GivensOrthogonalLinear
        else:
            layer_constructor = nn.Linear
        for i in range(n_layers):
            layers.append(layer_constructor(dims[i], dims[i + 1], bias=use_bias))
            if i < n_layers - 1:
                layers.append(nn.Tanh())
        self.mapping = nn.Sequential(*layers)
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapping(x)


class EncoderWithNHF(nn.Module):
    def __init__(
        self,
        main_encoder: Encoder,
        n_latents_per_pixel: int,
        use_orthogonal_layers: bool = False,
        num_orthogonal_layers: int = 3,
        use_bias_in_orthogonal_layers: bool = True,
        flatten_all_features: bool = False,
        disable_hashmap: bool = False,
    ):
        super().__init__()
        self.encoder = main_encoder
        cout = self.encoder.cout
        n_latent_dims = self.encoder.n_latent_dims
        # First, project cout to n_latents_per_pixel if needed
        if cout == n_latents_per_pixel:
            self.projection_layer = nn.Identity()
        else:
            self.projection_layer = nn.Linear(cout, n_latents_per_pixel)
            cout = n_latents_per_pixel
        # Now, if we flatten all features, the hashmap will be
        if disable_hashmap:
            self.hash_map = nn.Identity()
        else:
            if flatten_all_features:
                input_dim = n_latent_dims
                output_dim = n_latent_dims
            else:
                input_dim = cout
                output_dim = cout
            self.hash_map = DeepHashMap(
                input_dim=input_dim,
                output_dim=output_dim,
                n_layers=num_orthogonal_layers,
                use_orthogonal_layers=use_orthogonal_layers,
                use_bias=use_bias_in_orthogonal_layers,
            )
        self.flatten_all_features = flatten_all_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.projection_layer(x)
        batch_size = x.shape[0]
        h = w = x.shape[1]
        if self.flatten_all_features:
            x = rearrange(x, " b h w c -> b (h w c)")
        x = self.hash_map(x)
        if self.flatten_all_features:
            x = rearrange(x, "b (hz wz cz) -> b cz hz wz", b=batch_size, hz=h, wz=w)
        else:
            x = rearrange(x, " b h w c -> b c h w")
        return x
