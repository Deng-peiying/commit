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
    """Masked dual-frame IDM (shared-encoder, per-frame mask).

    Architecture:
        1. Shared UNet(3ch) generates per-frame soft masks:
             mask_t    = sigmoid(UNet(img_t))       → [B,1,H,W]
             mask_next = sigmoid(UNet(img_next))    → [B,1,H,W]
           Each mask precisely covers the arm position in its own frame.

        2. Shared RGB encoder + feature-level diff (RGB motion):
             feat_rgb_t    = RGB_enc(mask_t    * img_t)    → [B, FEAT_DIM]
             feat_rgb_next = RGB_enc(mask_next * img_next) → [B, FEAT_DIM]
             rgb_diff = feat_rgb_next - feat_rgb_t         → [B, FEAT_DIM]
           Masked RGB focuses on arm region; diff extracts semantic motion.

        3. Shared Depth encoder (global, no mask):
             feat_dep_t    = Dep_enc(dep_t)    → [B, DEP_DIM]
             feat_dep_next = Dep_enc(dep_next) → [B, DEP_DIM]
           - feat_dep_t doubles as global 3D spatial prior.
           - dep_diff = feat_dep_next - feat_dep_t → geometric motion.
           No mask: depth diff is naturally sparse; full scene geometry preserved.

        4. State encoder: MLP(pos_t) → [B, STATE_EMBED_DIM]

        5. Head(cat[rgb_diff, feat_dep_prior, dep_diff, feat_state]) → pos_{t+1}
    """

    DEP_DIM = 256

    def __init__(self, output_dim=OUTPUT_DIM):
        super().__init__()
        self.output_dim = output_dim

        # Shared mask generator: 3ch single-frame → soft foreground mask
        # Forward twice (once per frame) with shared weights.
        # 3 layers is sufficient for foreground segmentation, saves ~65M params.
        self.mask_net = UNet(in_channels=3, out_channels=1, base_channel=64, num_layers=3)

        # Shared RGB encoder: 3ch masked single-frame → FEAT_DIM
        self.rgb_encoder = ResNet(output_dim=FEAT_DIM, input_channels=3, resnet_type='34')

        # Shared Depth encoder: 1ch global depth → DEP_DIM
        # Used for both spatial prior (dep_t) and motion (dep_diff).
        self.dep_encoder = ResNet(output_dim=self.DEP_DIM, input_channels=1, resnet_type='34')

        # State encoder: pos_t → STATE_EMBED_DIM
        self.state_mlp = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, STATE_EMBED_DIM),
            nn.ReLU(inplace=True),
        )

        # Head: fused features → pos_{t+1}
        # rgb_diff(512) + dep_prior(256) + dep_diff(256) + state(128) = 1152
        head_in = FEAT_DIM + self.DEP_DIM + self.DEP_DIM + STATE_EMBED_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MaskedIDM (shared-enc, per-frame mask) — output_dim={output_dim}, "
              f"feat_rgb={FEAT_DIM}, dep={self.DEP_DIM}, state={STATE_EMBED_DIM}, "
              f"params={total:,}")

    def forward(self, img_t, dep_t, img_next, dep_next, pos_t, return_mask=False):
        if dep_t.dim() == 3:
            dep_t = dep_t.unsqueeze(1)
        if dep_next.dim() == 3:
            dep_next = dep_next.unsqueeze(1)

        # --- Per-frame hard mask (shared UNet, forward twice) ---
        # Straight-through estimator: hard 0/1 in forward, gradient flows through sigmoid in backward.
        soft_t    = torch.sigmoid(self.mask_net(img_t))      # [B, 1, H, W]
        soft_next = torch.sigmoid(self.mask_net(img_next))   # [B, 1, H, W]
        mask_t    = ((soft_t    > 0.5).float() - soft_t).detach() + soft_t     # hard 0/1
        mask_next = ((soft_next > 0.5).float() - soft_next).detach() + soft_next

        # --- RGB motion: shared encoder on masked frames, then diff ---
        feat_rgb_t    = self.rgb_encoder(mask_t    * img_t)    # [B, FEAT_DIM]
        feat_rgb_next = self.rgb_encoder(mask_next * img_next) # [B, FEAT_DIM]
        rgb_diff      = feat_rgb_next - feat_rgb_t             # [B, FEAT_DIM]

        # --- Depth: shared encoder, no mask ---
        feat_dep_t    = self.dep_encoder(dep_t)    # [B, DEP_DIM]  (also serves as spatial prior)
        feat_dep_next = self.dep_encoder(dep_next) # [B, DEP_DIM]
        dep_diff      = feat_dep_next - feat_dep_t # [B, DEP_DIM]  geometric motion

        # --- State embedding ---
        feat_state = self.state_mlp(pos_t)         # [B, STATE_EMBED_DIM]

        # --- Predict pos_{t+1} via residual: pos_t + delta ---
        # pos_t participates as feature (helps head understand current state),
        # but pos_t in the residual addition is detached to prevent the network
        # from short-circuiting (learning identity instead of delta).
        fused = torch.cat([rgb_diff, feat_dep_t, dep_diff, feat_state], dim=-1)
        delta = self.head(fused)                   # [B, output_dim]
        out   = pos_t.detach() + delta             # residual; detach prevents shortcut

        if return_mask:
            return out, (mask_t, mask_next)
        return out
