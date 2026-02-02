# test_hvit_phvit.py
import argparse
import math
import os

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

pi = math.pi

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))  # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0

    def HVIT(self, img):
        """
        img: (B,3,H,W), range [0,1]
        return: (B,3,H,W) with channels [H, V, I]
        """
        eps = 1e-8
        device = img.device
        dtype = img.dtype
        B, C, Hh, Ww = img.shape

        # value/max & min across channel
        value = img.max(1)[0].to(dtype)      # (B,H,W)
        img_min = img.min(1)[0].to(dtype)    # (B,H,W)

        # ---- FIX: use zeros, not uninitialized Tensor
        hue = torch.zeros((B, Hh, Ww), device=device, dtype=dtype)

        # compute hue (like HSV) via branch conditions
        # masks wrt max channel
        mR = (img[:, 0] == value)
        mG = (img[:, 1] == value)
        mB = (img[:, 2] == value)

        hue[mB] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[mB]
        hue[mG] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[mG]
        hue[mR] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[mR]) % 6.0

        # when min==value (gray), set hue=0
        hue[img_min == value] = 0.0
        hue = hue / 6.0  # normalize to [0,1)

        # saturation (HSV style)
        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)          # (B,1,H,W)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = float(k.item())

        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)  # (B,1,H,W)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        Hc = color_sensitive * saturation * ch
        Vc = color_sensitive * saturation * cv
        Ic = value
        xyz = torch.cat([Hc, Vc, Ic], dim=1)   # (B,3,H,W)
        return xyz

    def PHVIT(self, img):
        """
        img: (B,3,H,W) with channels [H, V, I]
        return: (B,3,H,W) reconstructed RGB in [0,1]
        """
        eps = 1e-8
        Hc, Vc, Ic = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # clip
        Hc = torch.clamp(Hc, -1, 1)
        Vc = torch.clamp(Vc, -1, 1)
        Ic = torch.clamp(Ic, 0, 1)

        v = Ic
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        Hn = Hc / (color_sensitive + eps)
        Vn = Vc / (color_sensitive + eps)
        Hn = torch.clamp(Hn, -1, 1)
        Vn = torch.clamp(Vn, -1, 1)

        # recover hue (0..1) and saturation (0..1)
        h = torch.atan2(Vn + eps, Hn + eps) / (2 * pi)
        h = h % 1.0
        s = torch.sqrt(Hn ** 2 + Vn ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        # HSV -> RGB (vectorized)
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1.0 - s)
        q = v * (1.0 - (f * s))
        t = v * (1.0 - ((1.0 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0], g[hi0], b[hi0] = v[hi0], t[hi0], p[hi0]
        r[hi1], g[hi1], b[hi1] = q[hi1], v[hi1], p[hi1]
        r[hi2], g[hi2], b[hi2] = p[hi2], v[hi2], t[hi2]
        r[hi3], g[hi3], b[hi3] = p[hi3], q[hi3], v[hi3]
        r[hi4], g[hi4], b[hi4] = t[hi4], p[hi4], v[hi4]
        r[hi5], g[hi5], b[hi5] = v[hi5], p[hi5], q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha

        return torch.clamp(rgb, 0.0, 1.0)

def load_image_as_tensor(path, device):
    """Load image as float tensor in [0,1], shape (1,3,H,W)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
    return tensor, img.size  # (W,H)

def tensor_to_image(t):
    """(1,3,H,W) in [0,1] -> PIL.Image"""
    t = t.detach().cpu().clamp(0, 1).squeeze(0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)

def mse(a, b):
    return torch.mean((a - b) ** 2)

def psnr_from_mse(m):
    if m <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / float(m))

def main():
    parser = argparse.ArgumentParser(description="Test HVIT -> PHVIT round-trip on an image.")
    parser.add_argument("image_path", type=str, help="Path to input image (jpg/png).")
    parser.add_argument("--save", action="store_true", help="Save reconstructed image and diff.")
    parser.add_argument("--show", action="store_true", help="Show images in a window.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 1) load image
    img_tensor, (W, H) = load_image_as_tensor(args.image_path, device)
    print(f"[Info] Loaded image size: {W}x{H}")

    # 2) init model
    model = RGB_HVI().to(device).eval()

    with torch.no_grad():
        # 3) HVIT
        xyz = model.HVIT(img_tensor)
        # 4) PHVIT
        recon = model.PHVIT(xyz)

        # 5) Metrics
        m = mse(img_tensor, recon).item()
        psnr = psnr_from_mse(m)
        print(f"[Result] MSE = {m:.8f}, PSNR = {psnr:.2f} dB, k = {model.this_k:.4f}")

    # 6) Optional: save/show
    if args.save or args.show:
        orig_img = tensor_to_image(img_tensor)
        recon_img = tensor_to_image(recon)
        # diff map (amplified for visibility)
        diff = (img_tensor - recon).abs().clamp(0, 1)
        diff_img = tensor_to_image((diff * 5.0).clamp(0, 1))  # amplify 5x

    if args.save:
        base, ext = os.path.splitext(args.image_path)
        out_recon = f"{base}_recon{ext}"
        out_diff = f"{base}_diff{ext}"
        recon_img.save(out_recon)
        diff_img.save(out_diff)
        print(f"[Saved] {out_recon}")
        print(f"[Saved] {out_diff}")

    if args.show:
        # Display using matplotlib
        o = np.array(orig_img)
        r = np.array(recon_img)
        d = np.array(diff_img)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(o); plt.title("Original"); plt.axis("off")
        plt.subplot(1, 3, 2); plt.imshow(r); plt.title("Reconstructed"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.imshow(d); plt.title("Abs Diff (x5)"); plt.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

#python test_hvit_phvit.py 1.png --save --show