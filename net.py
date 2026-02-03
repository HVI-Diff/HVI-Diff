import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *

# FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# WF_ROOT  = os.path.abspath(os.path.join(FILE_DIR, '..', 'WF-Diff-main'))
# sys.path.insert(0, WF_ROOT) if WF_ROOT not in sys.path else None

from net.wavelet import DWT, IWT






def make_fdense(nChannels, growthRate):
    return FDenseLayer(nChannels, growthRate)



class FDenseLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(FDenseLayer, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out



class SDAB(nn.Module):
    def __init__(self, nChannels, nDenselayer=3, growthRate=16):
        super(SDAB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(self._make_dense_layer(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def _make_dense_layer(self, nChannels, growthRate):
        layers = []
        layers.append(nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class PCCM(nn.Module):
    def __init__(self, nChannels, nDenselayer=1, growthRate=32):
        super(PCCM, self).__init__()
        nChannels_1 = nChannels
        nChannels_2 = nChannels
        modules1 = []
        for i in range(nDenselayer):
            modules1.append(make_fdense(nChannels_1, growthRate))
            nChannels_1 += growthRate
        self.dense_layers1 = nn.Sequential(*modules1)
        modules2 = []
        for i in range(nDenselayer):
            modules2.append(make_fdense(nChannels_2, growthRate))
            nChannels_2 += growthRate
        self.dense_layers2 = nn.Sequential(*modules2)
        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)
        self.SRDB = SRDB(nChannels)
        # self.patch_embed = PatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=nChannels,
        # embed_dim=embed_dims[0])

    def forward(self, x):
        x = self.SRDB(x)
        _, _, H, W = x.shape
        # print(x.shape)
        x_freq = torch.fft.rfft2(x, norm='backward')
        # print(x_freq.shape)
        mag = torch.abs(x_freq)
        # print(mag.shape)
        pha = torch.angle(x_freq)
        mag = self.dense_layers1(mag)
        # print(mag.shape)
        mag = self.conv_1(mag)
        # print(mag.shape)
        pha = self.dense_layers2(pha)
        pha = self.conv_2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        out = out + x
        return out


class CPB(nn.Module):
    def __init__(self, nChannels, growthRate=64):
        super(CPB, self).__init__()
        nChannels_ = nChannels
        modules1 = []
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
                               bias=False)
        self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
                               bias=False)
        self.conv3 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
                               bias=False)

        # self.conv11 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
        #                      bias=False)
        # self.conv22 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
        #                      bias=False)
        # self.conv33 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
        #                      bias=False)

        # self.conv4 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
        #                     bias=False)
        # self.conv5 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
        #                      bias=False)
        self.conv6 = nn.Conv2d(growthRate * 3, nChannels, kernel_size=1, padding=(1 - 1) // 2,
                               bias=False)
        self.leaky1 = nn.LeakyReLU(0.1, inplace=True)
        self.leaky2 = nn.LeakyReLU(0.1, inplace=True)
        self.leaky3 = nn.LeakyReLU(0.1, inplace=True)
        # self.bat1 = nn.BatchNorm2d(nChannels),
        # self.bat2 = nn.BatchNorm2d(nChannels),
        # self.bat3 = nn.BatchNorm2d(nChannels),
        # self.bat4 = nn.BatchNorm2d(nChannels),
        # self.bat5 = nn.BatchNorm2d(nChannels),
        # self.patch_embed = PatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=nChannels,
        # embed_dim=embed_dims[0])

    def forward(self, x):
        # x_1=self.bat1(self.conv1(x))
        x_1 = self.leaky1(self.conv1(x))
        x_2 = self.leaky2(self.conv2(x))
        x_3 = self.leaky3(self.conv3(x))
        x_0 = torch.cat((x_1, x_2, x_3), dim=1)
        # print(x_0.shape)

        # x_11=x_1+x_3
        # x_22=x_1+x_3+x_2
        # x_33=x_2+x_3

        # x_111= self.conv11(x_11)
        # x_222= self.conv22(x_22)
        # x_333= self.conv33(x_33)

        # x111=x_111*x_222+x_111
        # x333=x_222*x_333+x_333

        # x_o1= self.conv4(x111)
        # x_02= self.conv5(x333)

        # x_0=x_o1+x_02+x_1+x_3

        # x_0=self.conv6(x_0)
        # x_0=x_111+x+x_222+x_333
        x_0 = self.conv6(x_0)

        out = x_0 + x
        return out

class PDID(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()


        self.denoiser_hv_lf = nn.Sequential(
            nn.Conv2d(2, width, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(width, width, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(width, 2, 3, 1, 1, bias=False),
        )
        self.denoiser_hv_hf = nn.Sequential(
            nn.Conv2d(2, width, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(width, width, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(width, 2, 3, 1, 1, bias=False),
        )


        self.denoiser_i_lf = nn.Sequential(
            nn.Conv2d(1, width, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(width, width, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(width, 1, 3, 1, 1, bias=False),
        )
        self.denoiser_i_hf = nn.Sequential(
            nn.Conv2d(1, width, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(width, width, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(width, 1, 3, 1, 1, bias=False),
        )


        self.fusion = nn.Conv2d(3, 3, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        # x: [B,3,H,W] -> HV(2c) + I(1c)
        hv = x[:, :2, :, :]
        i  = x[:, 2:3, :, :]

        B = x.size(0)


        hv_dwt = self.dwt(hv)                 # [2B, 2, H/2, W/2]
        hv_lf, hv_hf = hv_dwt[:B], hv_dwt[B:]
        hv_lf = self.denoiser_hv_lf(hv_lf) + hv_lf
        hv_hf = self.denoiser_hv_hf(hv_hf) + hv_hf
        hv_restored = self.iwt(torch.cat([hv_lf, hv_hf], dim=0))  # [B,2,H,W]


        i_dwt = self.dwt(i)                   # [2B, 1, H/2, W/2]
        i_lf, i_hf = i_dwt[:B], i_dwt[B:]
        i_lf = self.denoiser_i_lf(i_lf) + i_lf
        i_hf = self.denoiser_i_hf(i_hf) + i_hf
        i_restored = self.iwt(torch.cat([i_lf, i_hf], dim=0))     # [B,1,H,W]


        out_hvi = torch.cat([hv_restored, i_restored], dim=1)     # [B,3,H,W]


        out = self.fusion(out_hvi) + x
        return out



class DIFFNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
                 ):
        super(DIFFNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        self.trans = RGB_HVI()

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi

pi = 3.141592653589793


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
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6

        hue[img.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I], dim=1)
        return xyz

    def PHVIT(self, img):
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb


