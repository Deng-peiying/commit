import torch
import torch.nn as nn

from .resnet import ResNet
from .unet import UNet

OUTPUT_DIM = 16   # left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
STATE_DIM = 16
FEAT_DIM = 512
STATE_EMBED_DIM = 128


class IDM(nn.Module):

    def __init__(self, model_name, output_dim=OUTPUT_DIM, *args, **kwargs):
        super().__init__()
        match model_name:
            case "mask":
                self.model = MaskedIDM(output_dim=output_dim, *args, **kwargs)
            case "resnet":
                self.model = ResNet(output_dim=output_dim, input_channels=4)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MaskedIDM(nn.Module):
    """Masked dual-frame IDM (concat, no explicit diff).

    Architecture:
        1. Shared UNet(3ch, 3 layers) generates per-frame hard masks:
             mask_t    = hard(sigmoid(UNet(img_t)))    → [B,1,H,W]
             mask_next = hard(sigmoid(UNet(img_next))) → [B,1,H,W]

        2. RGB encoder: concat masked frames → single forward
             feat_rgb = RGB_enc(cat[mask_t*img_t, mask_next*img_next])  → [B, FEAT_DIM]
           Network implicitly learns motion from 6ch concat (no explicit diff).
           Mask filters background; encoder focuses on arm region changes.

        3. Depth encoder: concat both depth frames → single forward
             feat_dep = Dep_enc(cat[dep_t, dep_next])  → [B, DEP_DIM]
           No mask. Contains both spatial prior (from dep_t) and geometric motion.

        4. State encoder: MLP(pos_t) → [B, STATE_EMBED_DIM]

        5. Head(cat[feat_rgb, feat_dep, feat_state]) → pos_{t+1}
    """

    DEP_DIM = 256

    def __init__(self, output_dim=OUTPUT_DIM):
        super().__init__()
        self.output_dim = output_dim

        # Shared mask generator: 3ch single-frame → hard foreground mask
        # Forward twice (once per frame) with shared weights.
        self.mask_net = UNet(in_channels=3, out_channels=1, base_channel=64, num_layers=3)

        # RGB encoder: 6ch (masked img_t + masked img_next) → FEAT_DIM
        self.rgb_encoder = ResNet(output_dim=FEAT_DIM, input_channels=6, resnet_type='34')

        # Depth encoder: 2ch (dep_t + dep_next) → DEP_DIM
        self.dep_encoder = ResNet(output_dim=self.DEP_DIM, input_channels=2, resnet_type='34')

        # State encoder: pos_t → STATE_EMBED_DIM
        self.state_mlp = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, STATE_EMBED_DIM),
            nn.ReLU(inplace=True),
        )

        # Head: fused features → pos_{t+1}
        # feat_rgb(512) + feat_dep(256) + state(128) = 896
        head_in = FEAT_DIM + self.DEP_DIM + STATE_EMBED_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MaskedIDM (concat, no diff) — output_dim={output_dim}, "
              f"feat_rgb={FEAT_DIM}, dep={self.DEP_DIM}, state={STATE_EMBED_DIM}, "
              f"params={total:,}")

    def forward(self, img_t, dep_t, img_next, dep_next, pos_t, return_mask=False):
        if dep_t.dim() == 3:
            dep_t = dep_t.unsqueeze(1)
        if dep_next.dim() == 3:
            dep_next = dep_next.unsqueeze(1)

        # --- Per-frame hard mask (shared UNet, forward twice) ---
        soft_t    = torch.sigmoid(self.mask_net(img_t))      # [B, 1, H, W]
        soft_next = torch.sigmoid(self.mask_net(img_next))   # [B, 1, H, W]
        mask_t    = ((soft_t    > 0.5).float() - soft_t).detach() + soft_t
        mask_next = ((soft_next > 0.5).float() - soft_next).detach() + soft_next

        # --- RGB: concat masked frames, single forward ---
        rgb_cat  = torch.cat([mask_t * img_t, mask_next * img_next], dim=1)  # [B, 6, H, W]
        feat_rgb = self.rgb_encoder(rgb_cat)                                  # [B, FEAT_DIM]

        # --- Depth: concat both frames, single forward, no mask ---
        dep_cat  = torch.cat([dep_t, dep_next], dim=1)  # [B, 2, H, W]
        feat_dep = self.dep_encoder(dep_cat)             # [B, DEP_DIM]

        # --- State embedding ---
        feat_state = self.state_mlp(pos_t)               # [B, STATE_EMBED_DIM]

        # --- Predict pos_{t+1} directly ---
        fused = torch.cat([feat_rgb, feat_dep, feat_state], dim=-1)
        out   = self.head(fused)                          # [B, output_dim]

        if return_mask:
            return out, (mask_t, mask_next)
        return out
