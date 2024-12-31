import logging
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from matplotlib import pyplot as plt
from typing import Dict, Tuple
import functools
from depth_pro.depth_pro import (
    create_model_and_transforms,
    create_backbone_model,
    DepthProConfig
)
from depth_pro.network.decoder import MultiresConvDecoder
from depth_pro.network.encoder import DepthProEncoder
from depth_pro.network.fov import FOVNetwork
from depth_pro.network.vit import resize_vit, resize_patch_embed
from depth_pro.utils import load_rgb

from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor
)



CONFIG_DICT: Dict[str, DepthProConfig] = {
    "large_192": DepthProConfig(
        patch_encoder_preset="dinov2l16_192",
        image_encoder_preset="dinov2l16_192",
        checkpoint_uri="./checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_192",
        encoder_scale_size=(192, 192),
        head_paddings=[1, 0, 1, 0],
        fov_head_paddings=[1, 2, 3, 0],
    ),
    "large_288": DepthProConfig(
        patch_encoder_preset="dinov2l16_288",
        image_encoder_preset="dinov2l16_288",
        checkpoint_uri="./checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_288",
        encoder_scale_size=(288, 288),
        head_paddings=[1, 0, 1, 0],
        fov_head_paddings=[1, 1, 2, 0],
    ),
    "large_384": DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri="./checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
        encoder_scale_size=(384, 384),
        head_paddings=[1, 0, 1, 0],
        fov_head_paddings=[1, 1, 1, 0],
    ),
}

class DepthDecoder(nn.Module):
    def __init__(self, head: nn.Module, fov: FOVNetwork, encoder_scale_size: (int, int)):
        super(DepthDecoder, self).__init__()
        self.head = head
        self.fov = fov
        self.encoder_scale_size = encoder_scale_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs[0]
        features = inputs[1]
        features_0 = inputs[2]

        # execute fov.forward locally with a different scale_factor
        # fov_deg = self.fov.forward(x, features_0.detach())
        if hasattr(self.fov, "encoder"):
            x = F.interpolate(
                x,
                size=self.encoder_scale_size,
                #scale_factor=self.encoder_scale_factor,
                mode="bilinear",
                align_corners=False,
            )
            x = self.fov.encoder(x)[:, 1:].permute(0, 2, 1)
            lowres_feature = self.fov.downsample(features_0.detach())
            x = x.reshape_as(lowres_feature) + lowres_feature
        else:
            x = features_0.detach()

        fov_deg = self.fov.head(x)
        
        canonical_inverse_depth = self.head(features)
        return canonical_inverse_depth, fov_deg

class DepthProScaled(nn.Module):
    def __init__(self, transform: nn.Module, encoder: DepthProEncoder, decoder: MultiresConvDecoder, depth: DepthDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = self.encoder(x)
        features, features_0 = self.decoder(encodings)
        depth, fov_deg = self.depth([x, features, features_0])
        return depth, fov_deg

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=False)
        return x

def create_scaled_model(config: DepthProConfig) -> DepthProScaled:
    patch_encoder, patch_encoder_config = create_backbone_model(preset = config.patch_encoder_preset)
    image_encoder, _ = create_backbone_model(preset = config.image_encoder_preset)
    fov_encoder, _ = create_backbone_model(preset = config.fov_encoder_preset)
    # fov_encoder = None

    dims_encoder = patch_encoder_config.encoder_feature_dims
    hook_block_ids = patch_encoder_config.encoder_feature_layer_ids
    encoder = DepthProEncoder(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=hook_block_ids,
        decoder_features=config.decoder_features,
    )

    decoder = MultiresConvDecoder(
        dims_encoder=[config.decoder_features] + list(encoder.dims_encoder),
        dim_decoder=config.decoder_features,
    )

    num_features = config.decoder_features
    fov = FOVNetwork(num_features=num_features, fov_encoder=fov_encoder)
    # Create FOV head.
    fov_head0 = [
        nn.Conv2d(
            num_features, num_features // 2, kernel_size=3, stride=2, padding=config.fov_head_paddings[0]
        ),  # 128 x 24 x 24
        nn.ReLU(True),
    ]
    fov_head = [
        nn.Conv2d(
            num_features // 2, num_features // 4, kernel_size=3, stride=2, padding=config.fov_head_paddings[1]
        ),  # 64 x 12 x 12
        nn.ReLU(True),
        nn.Conv2d(
            num_features // 4, num_features // 8, kernel_size=3, stride=2, padding=config.fov_head_paddings[2]
        ),  # 32 x 6 x 6
        nn.ReLU(True),
        nn.Conv2d(num_features // 8, 1, kernel_size=6, stride=1, padding=config.fov_head_paddings[3]),
    ]
    if fov_encoder is not None:
        fov.encoder = nn.Sequential(
            fov_encoder, nn.Linear(fov_encoder.embed_dim, num_features // 2)
        )
        fov.downsample = nn.Sequential(*fov_head0)
    else:
        fov_head = fov_head0 + fov_head
    fov.head = nn.Sequential(*fov_head)
    # fov = None

    last_dims = (32, 1)
    dim_decoder = config.decoder_features
    head = nn.Sequential(
        nn.Conv2d(
            dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=config.head_paddings[0]
        ),
        nn.ConvTranspose2d(
            in_channels=dim_decoder // 2,
            out_channels=dim_decoder // 2,
            kernel_size=2,
            stride=2,
            padding=config.head_paddings[1],
            bias=True,
        ),
        nn.Conv2d(
            dim_decoder // 2,
            last_dims[0],
            kernel_size=3,
            stride=1,
            padding=config.head_paddings[2],
        ),
        nn.ReLU(True),
        nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=config.head_paddings[3]),
        nn.ReLU(),
    )

    # Set the final convolution layer's bias to be 0.
    head[4].bias.data.fill_(0)

    # from depth_pro.py
    transform = nn.Sequential(
        #[
            #ToTensor(),
            #Lambda(lambda x: x.to(device)),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Interpolate(
                size=(encoder.img_size, encoder.img_size),
                mode="bilinear"
            ),
            ConvertImageDtype(torch.float32),
        #]
    )

    depth = DepthDecoder(head, fov, config.encoder_scale_size)
    load_state_dict(depth, config)

    model = DepthProScaled(transform, encoder, decoder, depth)
    load_state_dict(model, config)

    return model

def load_state_dict(model: nn.Module, config: DepthProConfig):
    checkpoint_uri = config.checkpoint_uri
    state_dict = torch.load(checkpoint_uri, map_location="cpu")
    _, _ = model.load_state_dict(
        state_dict=state_dict, strict=False
    )

def load_and_show_examples(models: tuple[DepthProScaled]):
    plt.ion()
    fig = plt.figure()
    ax_rgb = fig.add_subplot(1, 1 + len(models), 1)

    image, _, _ = load_rgb("data/example.jpg")
    ax_rgb.imshow(image)

    for index in range(len(models)):
        model_run = Compose([ToTensor(), Lambda(lambda x: x.to(torch.device("cpu"))), models[index]])
        depth_map = model_run(image).detach().cpu().numpy().squeeze()

        ax_disp = fig.add_subplot(1, 1 + len(models), 2 + index)
        ax_disp.imshow(depth_map, cmap="turbo")

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=True)





if __name__ == "__main__":
    model_192 = create_scaled_model(CONFIG_DICT["large_192"])
    model_288 = create_scaled_model(CONFIG_DICT["large_288"])
    model_384 = create_scaled_model(CONFIG_DICT["large_384"])

    # save_coreml_packages(model_192)
