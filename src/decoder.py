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

import torch
import torch.nn as nn
from einops import einsum, rearrange


class CNNDecoder(nn.Module):
    def __init__(
        self,
        shapez: tuple[int, int, int],
        factors: list[int],
        shapex: tuple[int, int, int],
        codebook_dim: int,
        n_layers_per_block: int = 3,
    ):
        super().__init__()
        self.shapez = shapez
        self.factors = factors
        self.shapex = shapex
        cz, hz, wz = shapez
        cx, hx, wx = shapex
        # Create a parameter with he normal initialization for the codebook:
        self.codebook = nn.Parameter(
            torch.randn(cz, codebook_dim, dtype=torch.float32) * (2 / codebook_dim) ** 0.5
        )
        cout = codebook_dim
        layers = []
        hi, wi = hz, wz
        for nl, factor in enumerate(factors):
            cin = cout
            cout = int(cout / factor)
            layers.append(nn.LayerNorm([cin, hi, wi]))
            hi = hi * 2
            wi = wi * 2
            # Expand the image dimension with same number of channels:
            layers.append(nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1))
            # layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm([cin, hi, wi]))
            # Decrease number of channels, keep image dimension (with convolutional transpose
            # layers):
            layers.append(nn.ConvTranspose2d(cin, cout, kernel_size=3, stride=1, padding=1))
            for _ in range(n_layers_per_block - 1):
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm([cout, hi, wi]))
                layers.append(nn.ConvTranspose2d(cout, cout, kernel_size=3, stride=1, padding=1))

        # Add a final layer with a bigger kernel size for better reconstruction
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm([cout, hi, wi]))
        layers.append(nn.ConvTranspose2d(cout, cout, kernel_size=7, stride=1, padding=3))

        # Add to the layers a resize to the final image shapex:
        self.deconv = nn.Sequential(*layers)
        self.final_resize = nn.UpsamplingNearest2d(size=(hx, wx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("Decoder input shape:", x.shape)
        # x is of shape (batch_size, n_samples, cz, hz, wz)
        batch_size = x.shape[0]
        x = rearrange(x, "b n cz hz wz -> (b n) cz hz wz")
        print("Rearranged decoder input shape:", x.shape)
        # Convert spin strings to bitstrings in {0, 1}:
        x = (x + 1) / 2
        # Now we perform a matrix multiplication with the codebook to get the
        # continuous representation:
        x = einsum(x, self.codebook, "b c h w, c d -> b d h w")
        print("After codebook multiplication shape:", x.shape)
        x = self.deconv(x)
        x = self.final_resize(x)
        x = rearrange(x, "(b n) c h w -> b n c h w", b=batch_size)
        return x
