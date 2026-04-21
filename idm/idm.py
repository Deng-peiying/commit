import torch
import torch.nn as nn

from .resnet import ResNet
from .unet import UNet

OUTPUT_DIM = 16   # left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
STATE_DIM = 16
VIS_DIM = 2048


class IDM(nn.Module):

    def __init__(self, model_name, output_dim=OUTPUT_DIM, *args, **kwargs):
        super().__init__()
        match model_name:
            case "mask":
                self.model = Mask(output_dim=output_dim, *args, **kwargs)
            case "resnet":
                self.model = ResNet(output_dim=output_dim, input_channels=8)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class Mask(nn.Module):
    """Dual-frame masked IDM that predicts action delta Δa = pos_{t+1} - pos_t.

    pos_t conditions the visual features via FiLM (γ * V_vis + β) rather than
    being concatenated directly, forcing the visual pathway to remain the primary
    source of information.
    """

    def __init__(self, output_dim=OUTPUT_DIM, *args, **kwargs):
        super().__init__()
        self.output_dim = output_dim

        # UNet: 8-ch → 1-ch spatial attention mask
        self.mask_model = UNet(in_channels=8, out_channels=1)

        # ResNet: 8-ch masked input → visual feature vector
        self.resnet_model = ResNet(output_dim=VIS_DIM, input_channels=8)

        # FiLM: pos_t → (γ, β), last layer zero-init so γ=β=0 at start
        self.film_mlp = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, VIS_DIM * 2),
        )
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        # Head: conditioned visual features → Δa
        self.head = nn.Sequential(
            nn.Linear(VIS_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Mask IDM (FiLM) — output_dim={output_dim}, params={total:,}")

    def forward(self, img_t, dep_t, img_next, dep_next, pos_t, return_mask=False):
        if dep_t.dim() == 3:
            dep_t = dep_t.unsqueeze(1)
        if dep_next.dim() == 3:
            dep_next = dep_next.unsqueeze(1)

        # 8-channel pair: [B, 8, H, W]
        pair = torch.cat([img_t, dep_t, img_next, dep_next], dim=1)

        # Spatial attention mask from dual-frame pair
        mask = (1 + torch.tanh(self.mask_model(pair))) / 2      # ∈ (0,1)
        mask_hard = (mask >= 0.5).float()
        masked = pair * ((mask_hard - mask).detach() + mask)    # straight-through

        # Visual features (primary pathway)
        v_vis = self.resnet_model(masked)                        # [B, VIS_DIM]

        # Residual FiLM: pos_t modulates visual features as a residual
        # V_out = (1 + γ) * V_vis + β
        # γ and β initialised to 0, so at init V_out = V_vis (visual path always active)
        film_params = self.film_mlp(pos_t)                       # [B, VIS_DIM*2]
        gamma, beta = film_params.chunk(2, dim=-1)               # each [B, VIS_DIM]
        v_conditioned = (1 + gamma) * v_vis + beta               # [B, VIS_DIM]

        delta = self.head(v_conditioned)                         # [B, output_dim]

        if return_mask:
            return delta, mask
        return delta
