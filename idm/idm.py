import torch
import torch.nn as nn

from .resnet import ResNet

OUTPUT_DIM = 16   # left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
STATE_DIM = 16
FEAT_DIM = 512
STATE_EMBED_DIM = 128


class IDM(nn.Module):

    def __init__(self, model_name, output_dim=OUTPUT_DIM, *args, **kwargs):
        super().__init__()
        match model_name:
            case "mask":
                self.model = DualFrameIDM(output_dim=output_dim, *args, **kwargs)
            case "resnet":
                self.model = ResNet(output_dim=output_dim, input_channels=4)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class DualFrameIDM(nn.Module):
    """Dual-frame IDM with shared-weight encoding (no mask).

    Architecture:
        1. Shared ResNet(4ch RGB-D) encodes each frame independently:
             feat_t    = ResNet([img_t, dep_t])      → [B, 512]
             feat_next = ResNet([img_next, dep_next]) → [B, 512]
        2. feat_diff = feat_next - feat_t             → [B, 512]  semantic-level motion
        3. v_state = MLP(pos_t)                       → [B, 128]  kinematic context
        4. Head([feat_diff, v_state])                 → pos_{t+1} [B, 16]

    Why no mask:
        - feat_diff already focuses on motion: static background cancels out
        - A single mask cannot cover both frames (arm is at different positions)
        - Fewer parameters, simpler training
    """

    def __init__(self, output_dim=OUTPUT_DIM, *args, **kwargs):
        super().__init__()
        self.output_dim = output_dim

        # Shared visual encoder: 4ch RGB-D → FEAT_DIM
        self.encoder = ResNet(output_dim=FEAT_DIM, input_channels=4)

        # State encoder: pos_t → STATE_EMBED_DIM
        self.state_mlp = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, STATE_EMBED_DIM),
            nn.ReLU(inplace=True),
        )

        # Head: (feat_diff + state) → pos_{t+1}
        self.head = nn.Sequential(
            nn.Linear(FEAT_DIM + STATE_EMBED_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DualFrameIDM (shared-encoder, no mask) — output_dim={output_dim}, "
              f"feat={FEAT_DIM}, state_embed={STATE_EMBED_DIM}, params={total:,}")

    def forward(self, img_t, dep_t, img_next, dep_next, pos_t, return_mask=False):
        if dep_t.dim() == 3:
            dep_t = dep_t.unsqueeze(1)
        if dep_next.dim() == 3:
            dep_next = dep_next.unsqueeze(1)

        # --- Shared encoder: each frame independently ---
        frame_t    = torch.cat([img_t, dep_t], dim=1)       # [B, 4, H, W]
        frame_next = torch.cat([img_next, dep_next], dim=1)  # [B, 4, H, W]

        feat_t    = self.encoder(frame_t)                    # [B, FEAT_DIM]
        feat_next = self.encoder(frame_next)                 # [B, FEAT_DIM]

        # --- Motion feature (semantic-level diff) ---
        feat_diff = feat_next - feat_t                        # [B, FEAT_DIM]

        # --- State embedding ---
        v_state = self.state_mlp(pos_t)                       # [B, STATE_EMBED_DIM]

        # --- Predict pos_{t+1} ---
        out = self.head(torch.cat([feat_diff, v_state], dim=-1))  # [B, output_dim]

        if return_mask:
            return out, None  # no mask, but keep interface compatible
        return out
