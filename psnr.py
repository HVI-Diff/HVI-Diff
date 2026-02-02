import os
import cv2
import torch
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(folder1, folder2):
    # 初始化 LPIPS（使用 VGG 网络）
    loss_fn = lpips.LPIPS(net='vgg').to("cuda" if torch.cuda.is_available() else "cpu")

    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    psnr_list, ssim_list, lpips_list = [], [], []

    for f1, f2 in zip(files1, files2):
        path1 = os.path.join(folder1, f1)
        path2 = os.path.join(folder2, f2)

        if not (path1.lower().endswith(('png','jpg','jpeg')) and path2.lower().endswith(('png','jpg','jpeg'))):
            continue

        # 读取图像
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            print(f"跳过：{f1}, {f2}，图像读取失败")
            continue

        # 确保大小一致
        if img1.shape != img2.shape:
            print(f"跳过：{f1}, {f2}，尺寸不一致 {img1.shape} vs {img2.shape}")
            continue

        # 转换为RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # ---- PSNR ----
        psnr_val = psnr(img1, img2, data_range=255)

        # ---- SSIM ----
        ssim_val = ssim(img1, img2, channel_axis=-1, data_range=255)

        # ---- LPIPS ----
        t1 = torch.tensor(img1/255.0).permute(2,0,1).unsqueeze(0).float()
        t2 = torch.tensor(img2/255.0).permute(2,0,1).unsqueeze(0).float()
        if torch.cuda.is_available():
            t1, t2 = t1.cuda(), t2.cuda()
        lpips_val = loss_fn(t1, t2).item()

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)

        print(f"{f1} vs {f2} -> PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")

    print("\n=== 平均结果 ===")
    print(f"PSNR: {np.mean(psnr_list):.4f}")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
    print(f"LPIPS: {np.mean(lpips_list):.4f}")

if __name__ == "__main__":
    folder1 = "/home/chp/HVI-CIDNet-master/datasets/LOLdataset/eval15/high"   # 原始图像
    folder2 = "/home/chp/HVI-CIDNet-master/output/LOLv1"   # 恢复/生成图像
    calculate_metrics(folder1, folder2)
