import torch
import numpy as np
import torchvision
from PIL import Image


class DinoPreprocessor:
    """Handles image and depth preprocessing for dual-frame IDM."""

    AUG_CONFIG = {
        'brightness_range': ((0.8, 1.2), (0.4, 1.6)),
        'contrast_range': ((0.7, 1.3), (0.7, 1.7)),
        'saturation_range': ((0.5, 1.5), (0.5, 2.0)),
        'hue_shift': (0.05, 0.10),
        'random_apply_prob': (0.4, 0.8),
        'sharpness_factor': (1.8, 2.2),
        'sharpness_prob': (0.7, 0.9)
    }

    def __init__(self, args):
        self.use_transform = args.use_transform
        self.current_progress = 0.0
        self.height = 720
        self.width = 640
        self.dino_size = 518
        self._build_transforms()

    def _lerp(self, start, end, progress):
        return start + (end - start) * progress

    def _build_transforms(self):
        p = self.current_progress
        params = {}
        for key, (start, end) in self.AUG_CONFIG.items():
            if isinstance(start, tuple):
                params[key] = (
                    self._lerp(start[0], end[0], p),
                    self._lerp(start[1], end[1], p),
                )
            else:
                params[key] = self._lerp(start, end, p)

        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=params['brightness_range'],
            contrast=params['contrast_range'],
            saturation=params['saturation_range'],
            hue=params['hue_shift'],
        )
        self.resize_rgb = torchvision.transforms.Resize((self.dino_size, self.dino_size))
        self.resize_depth = torchvision.transforms.Resize(
            (self.dino_size, self.dino_size),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        self.normalize_rgb = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def set_augmentation_progress(self, progress):
        self.current_progress = max(0.0, min(1.0, progress))
        self._build_transforms()

    # ------------------------------------------------------------------
    # single-item processors
    # ------------------------------------------------------------------

    def _apply_jitter(self, image, jitter_params):
        """Apply pre-sampled ColorJitter params (fn_idx, b, c, s, h) to a PIL image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        fn_idx, brightness, contrast, saturation, hue = jitter_params
        tf = torchvision.transforms.functional
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                image = tf.adjust_brightness(image, brightness)
            elif fn_id == 1 and contrast is not None:
                image = tf.adjust_contrast(image, contrast)
            elif fn_id == 2 and saturation is not None:
                image = tf.adjust_saturation(image, saturation)
            elif fn_id == 3 and hue is not None:
                image = tf.adjust_hue(image, hue)
        return image

    def _to_tensor_rgb(self, image):
        """PIL RGB (possibly already jittered) → normalised [3, H, W] tensor."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        t = torchvision.transforms.functional.to_tensor(image).float()
        t = self.resize_rgb(t)
        return self.normalize_rgb(t)

    def process_image(self, image):
        """PIL RGB → normalised [3, dino_size, dino_size] tensor (single-frame path)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if self.use_transform:
            image = self.color_jitter(image)
        return self._to_tensor_rgb(image)

    def process_depth(self, depth):
        """PIL L-mode depth → normalised [1, dino_size, dino_size] tensor."""
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth, mode='L')
        t = torchvision.transforms.functional.to_tensor(depth).float()  # [1, H, W], 0-1
        t = self.resize_depth(t)
        # Standardise to zero-mean unit-variance using uint8 midpoint
        t = (t - 0.5) / 0.5
        return t

    # ------------------------------------------------------------------
    # pair processor (main entry point for the dataset)
    # ------------------------------------------------------------------

    def process_pair(self, img_t, dep_t, img_next, dep_next, pos_t, pos_next):
        """Process a (t, t+1) frame pair.

        Both RGB frames share the same ColorJitter parameters so the UNet
        sees only real motion, not augmentation-induced colour differences.
        Depth is never colour-jittered.
        """
        if self.use_transform:
            # Sample one fixed jitter transform and apply it to both frames
            jitter_fn = self.color_jitter.get_params(
                self.color_jitter.brightness,
                self.color_jitter.contrast,
                self.color_jitter.saturation,
                self.color_jitter.hue,
            )
            img_t    = self._apply_jitter(img_t,    jitter_fn)
            img_next = self._apply_jitter(img_next, jitter_fn)

        img_t    = self._to_tensor_rgb(img_t)
        img_next = self._to_tensor_rgb(img_next)
        dep_t    = self.process_depth(dep_t)
        dep_next = self.process_depth(dep_next)

        if self.use_transform and np.random.random() < 0.5:
            img_t, dep_t, img_next, dep_next, pos_t, pos_next = \
                self._flip_pair(img_t, dep_t, img_next, dep_next, pos_t, pos_next)

        return img_t, dep_t, img_next, dep_next, pos_t, pos_next

    # ------------------------------------------------------------------
    # flip helpers
    # ------------------------------------------------------------------

    def _flip_tensor(self, t):
        """Horizontal flip of [C, H, W] tensor."""
        return torch.flip(t, dims=[2])

    def _flip_pos(self, pos):
        """Mirror 16-dim qpos: swap left/right halves, negate lateral axes.

        Layout: [left_arm(7), left_grip(1), right_arm(7), right_grip(1)]
        """
        flipped = torch.zeros_like(pos)
        flipped[:8] = pos[8:]
        flipped[8:] = pos[:8]
        # Negate x-translation and y/z rotation components for each arm (indices 0,4,5 within each 7-dim arm)
        for offset in [0, 8]:
            flipped[offset + 0] *= -1
            flipped[offset + 4] *= -1
            flipped[offset + 5] *= -1
        return flipped

    def _flip_pair(self, img_t, dep_t, img_next, dep_next, pos_t, pos_next):
        return (
            self._flip_tensor(img_t),
            self._flip_tensor(dep_t),
            self._flip_tensor(img_next),
            self._flip_tensor(dep_next),
            self._flip_pos(pos_t),
            self._flip_pos(pos_next),
        )

    # kept for backward-compatibility with any external callers
    def handle_flip(self, images, pos):
        if isinstance(images, Image.Image):
            images = torchvision.transforms.functional.to_tensor(images).float()
        return self._flip_tensor(images), self._flip_pos(pos)
