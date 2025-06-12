# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 12:18:53 2025

@author: sunnycyc
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

class SpectralFlux(torch.nn.Module):
    def __init__(self, N_bands=8, gamma_init=10.0, N_differentiation=5, a_lrelu=0.0, N_local_average=11,
                 N_gaussian=15, sigma_gaussian=3, gamma_trainable=False, diff_trainable=False, loc_avg_trainable=False,
                 weighted_sum_trainable=False, gaussian_trainable=False, eps = 1e-8
                 ):
        super().__init__()

        self.N_bands = N_bands
        self.eps = eps
        # logarithmic compression
        self.log_gamma = torch.nn.Parameter(torch.ones(N_bands) * torch.log(torch.as_tensor(gamma_init)), requires_grad=gamma_trainable)

        # differentiation
        self.N_differentiation = N_differentiation

        self.diff = torch.nn.Conv2d(
            in_channels=self.N_bands,
            out_channels=self.N_bands,
            kernel_size=(1, N_differentiation),
            stride=1,
            padding="same",
            groups=self.N_bands,
        )

        diff_kernel = torch.zeros(self.N_bands, 1, 1, N_differentiation)
        diff_kernel[..., N_differentiation // 2] = -1
        diff_kernel[..., N_differentiation // 2 + 1] = 1
        self.diff.weight.data = diff_kernel

        if not diff_trainable:
            for param in self.diff.parameters():
                param.requires_grad = False

        # half-wave rectification
        self.half_wave_rect = torch.nn.LeakyReLU(negative_slope=a_lrelu)

        # local averaging
        self.local_average = torch.nn.Conv1d(
            in_channels=8,
            out_channels=8,
            kernel_size=N_local_average,
            stride=1,
            padding="same",
            groups=self.N_bands
        )

        self.local_average.weight.data = torch.ones(N_bands, 1, N_local_average) / N_local_average

        if not loc_avg_trainable:
            for param in self.local_average.parameters():
                param.requires_grad = False

        # combine bands (weighted sum)
        self.mix_chunks = torch.nn.Linear(
            in_features=self.N_bands, 
            out_features=1,
            bias=False,
        )

        self.mix_chunks.weight.data = torch.ones(1, N_bands) / N_bands

        if not weighted_sum_trainable:
            for param in self.mix_chunks.parameters():
                param.requires_grad = False

        # gaussian smoothing
        self.gaussian_smoothing = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=N_gaussian,
            stride=1,
            padding="same",
        )

        gaussian_window = torch.exp(-(torch.arange(N_gaussian) - N_gaussian // 2) ** 2 / (2 * sigma_gaussian))
        self.gaussian_smoothing.weight.data = gaussian_window.view(1, 1, -1)

        if not gaussian_trainable:
            for param in self.gaussian_smoothing.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x is of shape (batch, freq, time)
        out = {}

        # split in bands
        x = torch.chunk(x, chunks=self.N_bands, dim=1)  
        x = torch.stack(x, dim=3)  # (batch, freq, time, chunk)

        # logarithmic compression
        x = torch.log(1 + torch.exp(self.log_gamma) * x)
        out["x_log"] = x.clone()

        # differentiation
        x = x.permute(0, 3, 1, 2)  # (batch, chunk, freq, time)
        x = self.diff(x)
        out["x_diff"] = x.clone()

        # half-wave rectification
        x = self.half_wave_rect(x)  # (batch, chunk, freq, time)
        out["x_half_wave"] = x.clone()

        # pooling over frequency axis
        x = x.sum(dim=2)  # (batch, chunk, time)
        out["x_pool"] = x.clone()

        # local average subtraction
        x_local_avg = self.local_average(x)
        x = x - x_local_avg  # (batch, chunk, time)
        out["x_local_avg"] = x.clone()

        # weighted sum of bands
        x = self.mix_chunks(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, 1, time)
        out["x_weighted_sum"] = x.clone()

        x = self.gaussian_smoothing(x).squeeze(dim=1)  # (batch, time)
        out["x_gaussian"] = x.clone()

        # global max normalization
        x = self.half_wave_rect(x)
        # x = x / x.max(dim=1, keepdim=True).values  # (batch, time)
        # Small constant to prevent division by zero, eps = 1e-8
        x = x / (x.max(dim=1, keepdim=True).values + self.eps)
        out["x_act"] = x

        return out

class DifferentiableTempogramFourier(torch.nn.Module):
    def __init__(self, Fs_nov, N, H, Theta=None, train_tempogram = False):
        """
        Compute Fourier-based tempogram in a differentiable manner.

        Args:
            Fs_nov (float): Sampling rate
            N (int): Window length
            H (int): Hop size
            Theta (torch.Tensor): Set of tempi (given in BPM)
        """
        super().__init__()

        self.Fs_nov = Fs_nov
        self.N = N
        self.H = H
        self.Theta = Theta if Theta is not None else torch.arange(30, 601, 1)

        exp_all = torch.zeros(self.Theta.numel(), self.N, dtype=torch.complex64)

        for k, theta in enumerate(self.Theta):
            t = torch.arange(self.N)
            omega = (theta / 60.0) / self.Fs_nov
            exp_all[k, :] = torch.exp(-2 * torch.pi * 1j * omega * t)  # Shape: (L_pad)

        exp_all = torch.flip(exp_all, (-1,)) * torch.hann_window(N)

        self.compute_tempogram = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.Theta.numel(),
            kernel_size=self.N,
            stride=H,
            bias=False,
        )

        self.compute_tempogram.weight.data = exp_all.unsqueeze(dim=1)
        if not train_tempogram:
            for param in self.compute_tempogram.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input signal of shape (batch_size, num_samples)

        Returns:
            X (torch.Tensor): Tempogram of shape (batch_size, len(Theta), M)
            T_coef (torch.Tensor): Time axis (seconds)
            F_coef_BPM (torch.Tensor): Tempo axis (BPM)
        """
        x = F.pad(x, (self.N // 2, self.N // 2))
        X = self.compute_tempogram(x.unsqueeze(dim=1).type(torch.complex64))
        M = X.shape[-1]

        # Time and frequency coefficients
        T_coef = torch.arange(M, device=x.device).float() * self.H / self.Fs_nov
        F_coef_BPM = self.Theta

        return X, T_coef, F_coef_BPM

class ComputePLP(torch.nn.Module):
    def __init__(
            self, 
            Fs_nov, 
            N, 
            H, 
            Theta=None,
            temp=1,
            plp_mode="softmax",
            train_plp = False, 
        ):
        """
        Compute Fourier-based tempogram in a differentiable manner.

        Args:
            Fs_nov (float): Sampling rate
            N (int): Window length
            H (int): Hop size
            Theta (torch.Tensor): Set of tempi (given in BPM)
            temp (float): Temperature for softmax normalization
            plp_mode (str): PLP mode ('softmax' or 'argmax')
        """
        super().__init__()

        self.Fs_nov = Fs_nov
        self.N = N
        self.H = H
        self.Theta = Theta if Theta is not None else torch.arange(30, 601, 1)
        self.temp = temp
        self.plp_mode = plp_mode

        exp_all = torch.zeros(self.Theta.numel(), self.N, dtype=torch.complex64)

        for k, theta in enumerate(self.Theta):
            t = torch.arange(self.N)
            omega = (theta / 60.0) / self.Fs_nov
            exp_all[k, :] = torch.exp(2 * torch.pi * 1j * omega * t)  # Shape: (L_pad)

        window = torch.hann_window(N) 
        window = window / window.sum() * H
        exp_all = torch.flip(exp_all, (-1,)) * window

        self.compute_plp = torch.nn.ConvTranspose1d(
            in_channels=self.Theta.numel(),
            out_channels=1,
            kernel_size=self.N,
            stride=H,
            bias=False,
        )

        self.compute_plp.weight.data = exp_all.unsqueeze(dim=1)
        if not train_plp:
            for param in self.compute_plp.parameters():
                param.requires_grad = False
            

    def forward(self, X):
        """
        Args:
            x (torch.Tensor): Input signal of shape (batch_size, num_samples)

        Returns:
            X (torch.Tensor): Tempogram of shape (batch_size, len(Theta), M)
            T_coef (torch.Tensor): Time axis (seconds)
            F_coef_BPM (torch.Tensor): Tempo axis (BPM)
        """
        if self.plp_mode == "softmax":
            coefs = F.softmax(torch.abs(X / self.temp), dim=1)
        elif self.plp_mode == "argmax":
            max_indices = torch.argmax(torch.abs(X), dim=1)
            coefs = F.one_hot(max_indices, num_classes=X.shape[1]).transpose(1, 2).float()
        else:
            raise ValueError(f"Unknown plp_mode: {self.plp_mode}")

        phase = X / (X.abs() + 1e-8)

        x = self.compute_plp(coefs * phase)
        x = torch.real(x)
        # x = x[..., 0, self.N // 2 : (-self.N // 2)]
        x = x[..., 0, self.N // 2 : (self.N // 2)+ self.H*X.shape[-1]]

        # half-wave rect
        x = F.relu(x)
        return x

class Novelty2PLP(nn.Module):
    def __init__(self, Fs_nov, N, H, Theta, temp=1, plp_mode='softmax', 
                ):
        """
        Compute windowed sinusoid with optimal phase as a trainable PyTorch module.

        Args:
            Fs (float): Sampling rate.
            N (int): Window length.
            H (int): Hop size.
            Theta (torch.Tensor): Set of tempi (given in BPM).
            temp (float): Temperature for softmax normalization.
            plp_mode (str): PLP mode ('softmax' or 'equal').
            return_nonrect (bool): If True, return non-rectified PLP.
        """
        super(Novelty2PLP, self).__init__()
        self.Fs_nov = Fs_nov
        self.N = N
        self.H = H
        self.Theta = Theta
        # self.H_tmp = H_tmp
        # self.H_plp = H_plp
        # self.fixplpwin = fix_plpwin
        self.temp = temp
        self.plp_mode = plp_mode
        # self.return_nonrect = return_nonrect
        self.dTempogram = DifferentiableTempogramFourier(self.Fs_nov, 
                                                         self.N, 
                                                         self.H, Theta=self.Theta)

        self.dPLP = ComputePLP(self.Fs_nov, 
                                   self.N, self.H, self.Theta, self.temp, 
                                   self.plp_mode, 
                                   )
    def forward(self, nov):
        
        X, T_coef, F_coef_BPM = self.dTempogram(nov)
        nov_PLP_rect = self.dPLP(X)
        return nov_PLP_rect

class dPLPM1(nn.Module):
    def __init__(self, Fs_nov, H, Theta, temp=1, plp_mode='softmax', 
                 return_nonrect=False,
                 # freq_dim = 1024, 
                 N_bands=8, gamma_init=10, N_differentiation=5, 
                 a_lrelu=0.0, N_local_average=11,
                 N_gaussian=15, sigma_gaussian=3,
                 gamma_trainable=True, 
                 diff_trainable = True,  loc_avg_trainable=False,
                 weighted_sum_trainable=True, gaussian_trainable=False,
                 N10 = True, N5 = True, N3 = True, fuser_fc_trainable = True, 
                 fuser_gs_trainable = False, fuser_N_gaussian = 15, 
                 fuser_sigma_gaussian = 3,
                                  ):
        """
        Compute windowed sinusoid with optimal phase as a trainable PyTorch module.

        Args:
            Fs (float): Sampling rate.
            N (int): Window length.
            H (int): Hop size.
            Theta (torch.Tensor): Set of tempi (given in BPM).
            temp (float): Temperature for softmax normalization.
            plp_mode (str): PLP mode ('softmax' or 'equal').
            return_nonrect (bool): If True, return non-rectified PLP.
        """
        super(dPLPM1, self).__init__()
        self.Fs_nov = Fs_nov
        # self.N = N
        self.H = H
        self.Theta = Theta
        self.temp = temp
        self.plp_mode = plp_mode
        # self.return_nonrect = return_nonrect
        self.N10 = N10
        self.N5 = N5
        self.N3 = N3
        ### for spectral flux
        self.N_bands = N_bands
        self.gamma_init = gamma_init
        self.N_differentiation = N_differentiation
        self.a_lrelu = a_lrelu
        self.N_local_average = N_local_average
        self.N_gaussian = N_gaussian
        self.sigma_gaussian = sigma_gaussian
        self.gamma_trainable = gamma_trainable
        self.diff_trainable = diff_trainable
        self.loc_avg_trainable = loc_avg_trainable
        self.weighted_sum_trainable = weighted_sum_trainable
        self.gaussian_trainable = gaussian_trainable
        ### for fuser
        self.N_acti = sum([N10, N5, N3]) + 1
        self.fuser_fc_trainable = fuser_fc_trainable
        self.fuser_sigma_gaussian = fuser_sigma_gaussian
        self.fuser_gs_trainable = fuser_gs_trainable
        self.fuser_N_gaussian = fuser_N_gaussian
        
        if self.N3:
            self.nov2plpn3 = Novelty2PLP(Fs_nov, N=300, H = H, Theta = Theta, 
                                         temp=1, plp_mode='softmax', 
                                         )
        else:
            self.nov2plpn3 = None
        if self.N5:
            self.nov2plpn5 = Novelty2PLP(Fs_nov, N=500, H = H, Theta = Theta, 
                                         temp=1, plp_mode='softmax', )
        else:
            self.nov2plpn5 = None
        if self.N10:
            self.nov2plpn10 = Novelty2PLP(Fs_nov, N=1000, H = H, Theta = Theta, 
                                         temp=1, plp_mode='softmax', 
                                         )
        else:
            self.nov2plpn10 = None

        self.NovNet = SpectralFlux(N_bands=self.N_bands, gamma_init=self.gamma_init, 
                                   N_differentiation= self.N_differentiation, 
                                   a_lrelu= self.a_lrelu, 
                                   N_local_average= self.N_local_average,
                                   N_gaussian = self.N_gaussian, 
                                   sigma_gaussian= self.sigma_gaussian,
                                   gamma_trainable= self.gamma_trainable, 
                                   diff_trainable= self.diff_trainable ,  
                                   loc_avg_trainable= self.loc_avg_trainable,
                                   weighted_sum_trainable= self.weighted_sum_trainable, 
                                   gaussian_trainable= self.gaussian_trainable)
        
        self.Fuser = Fuser(N_act = self.N_acti, 
                           fc_trainable = self.fuser_fc_trainable, 
                           N_gaussian = self.fuser_N_gaussian, 
                           gaussian_trainable= self.fuser_gs_trainable, 
                           sigma_gaussian= self.fuser_sigma_gaussian)
            
    def forward(self, x):
        # x shape (batch, 1024, time)
        out = {}
        x = self.NovNet(x) # (batch, time)
        spectral_flux = x["x_act"]
        out['spectral_flux'] = spectral_flux.clone()
        # print('x-act shape:{}'.format(x["x_act"].shape))
        
        nov_plp_list = []
        mapping = {
            "N10": (self.N10, self.nov2plpn10),
            "N5": (self.N5, self.nov2plpn5),
            "N3": (self.N3, self.nov2plpn3),
        }
        
        for key, (flag, func) in mapping.items():
            if flag:
                plp_tmp = func(spectral_flux)
                nov_plp_list.append(plp_tmp.unsqueeze(-1))
                out[key] = plp_tmp.clone()
                
        # nov_plp_concat  shape (batch, time, num of plps)
        nov_plp_concat = torch.cat(nov_plp_list, dim=2) if nov_plp_list else None
        # print('nov_plp_cat shape:{}'.format(nov_plp_concat.shape))

        
        concatenated = torch.cat((spectral_flux.unsqueeze(-1), nov_plp_concat[:, :spectral_flux.shape[1]]), dim=-1)
        # print(concatenated.shape)
        fused_out = self.Fuser(concatenated)
        fused_nov = fused_out["x_act"]
        out['fused_nov'] = fused_nov.clone() 
        # print('grad, plp:{}, spectral_flux:{}, fused_nov:{}'.format(nov_plp_concat.grad, 
        #                                                             spectral_flux.grad, fused_nov.grad))
        
        return out

    
class dPLPM3(nn.Module):
    def __init__(self, Fs_nov, H, temp=1, plp_mode='softmax', 
                 # freq_dim = 1024, 
                 N_bands=8, gamma_init=10, N_differentiation=5, 
                 a_lrelu=0.0, N_local_average=11,
                 N_gaussian=15, sigma_gaussian=3,
                 gamma_trainable=True, 
                 diff_trainable = True,  loc_avg_trainable=False,
                 weighted_sum_trainable=True, gaussian_trainable=False,
                 fuser_fc_trainable = True, 
                 fuser_gs_trainable = False, fuser_N_gaussian = 15, 
                 fuser_sigma_gaussian = 3,
                                  ):
        """
        Compute windowed sinusoid with optimal phase as a trainable PyTorch module.

        Args:
            Fs (float): Sampling rate.
            N (int): Window length.
            H (int): Hop size.
            # Theta (torch.Tensor): Set of tempi (given in BPM).
            temp (float): Temperature for softmax normalization.
            plp_mode (str): PLP mode ('softmax' or 'equal').

        """
        super(dPLPM3, self).__init__()
        self.Fs_nov = Fs_nov
        # self.N = N
        self.H = H
        # self.Theta = Theta
        self.temp = temp
        self.plp_mode = plp_mode

        ### for spectral flux
        self.N_bands = N_bands
        self.gamma_init = gamma_init
        self.N_differentiation = N_differentiation
        self.a_lrelu = a_lrelu
        self.N_local_average = N_local_average
        self.N_gaussian = N_gaussian
        self.sigma_gaussian = sigma_gaussian
        self.gamma_trainable = gamma_trainable
        self.diff_trainable = diff_trainable
        self.loc_avg_trainable = loc_avg_trainable
        self.weighted_sum_trainable = weighted_sum_trainable
        self.gaussian_trainable = gaussian_trainable
        ### for fuser
        self.N_acti = 4
        self.fuser_fc_trainable = fuser_fc_trainable
        self.fuser_sigma_gaussian = fuser_sigma_gaussian
        self.fuser_gs_trainable = fuser_gs_trainable
        self.fuser_N_gaussian = fuser_N_gaussian

        self.NovNet = SpectralFlux(N_bands=self.N_bands, gamma_init=self.gamma_init, 
                                   N_differentiation= self.N_differentiation, 
                                   a_lrelu= self.a_lrelu, 
                                   N_local_average= self.N_local_average,
                                   N_gaussian = self.N_gaussian, 
                                   sigma_gaussian= self.sigma_gaussian,
                                   gamma_trainable= self.gamma_trainable, 
                                   diff_trainable= self.diff_trainable ,  
                                   loc_avg_trainable= self.loc_avg_trainable,
                                   weighted_sum_trainable= self.weighted_sum_trainable, 
                                   gaussian_trainable= self.gaussian_trainable)
        
        self.Fuser = Fuser(N_act = self.N_acti, 
                           fc_trainable = self.fuser_fc_trainable, 
                           N_gaussian = self.fuser_N_gaussian, 
                           gaussian_trainable= self.fuser_gs_trainable, 
                           sigma_gaussian= self.fuser_sigma_gaussian)
            
    def forward(self, x):
        # x shape (batch, 1024, time)
        out = {}
        x = self.NovNet(x) # (batch, time)
        spectral_flux = x["x_act"]
        out['spectral_flux'] = spectral_flux.clone()
        # print('x-act shape:{}'.format(x["x_act"].shape))
        
        nov_plp_list = [spectral_flux.unsqueeze(-1), 
                        spectral_flux.unsqueeze(-1), 
                        spectral_flux.unsqueeze(-1)]

        nov_plp_concat = torch.cat(nov_plp_list, dim=2) if nov_plp_list else None
        # print('nov_plp_cat shape:{}'.format(nov_plp_concat.shape))

        
        concatenated = torch.cat((spectral_flux.unsqueeze(-1), nov_plp_concat[:, :spectral_flux.shape[1]]), dim=-1)
        # print(concatenated.shape)
        fused_out = self.Fuser(concatenated)
        fused_nov = fused_out["x_act"]
        out['fused_nov'] = fused_nov.clone() 
        
        return out

class Fuser(torch.nn.Module):
    def __init__(self, N_act=4, fc_trainable = False, N_gaussian=15, gaussian_trainable=True, 
                 sigma_gaussian=3, 
                 # a_lrelu = 0.0, 
                 eps = 1e-8):
        super().__init__()
        # number of activation functions to fuse
        self.N_act = N_act
        self.eps = eps
        # Weight sum the curves
        self.fc = torch.nn.Linear(
            in_features=self.N_act, 
            out_features=1,
            bias=True,
        )
        self.fc.weight.data = torch.ones(1, N_act) / N_act
        if not fc_trainable:
            for param in self.fc.parameters():
                param.requires_grad = False
        
        ### Gaussian smoothing
        self.gaussian_smoothing = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=N_gaussian,
            stride=1,
            padding="same",
        )

        gaussian_window = torch.exp(-(torch.arange(N_gaussian) - N_gaussian // 2) ** 2 / (2 * sigma_gaussian))
        self.gaussian_smoothing.weight.data = gaussian_window.view(1, 1, -1)
        if not gaussian_trainable:
            for param in self.gaussian_smoothing.parameters():
                param.requires_grad = False
                
        # recfification and normalize to 0~1
        self.rectification = torch.nn.Sigmoid()
        # self.rectification = torch.nn.LeakyReLU(negative_slope=a_lrelu)
        
    def forward(self, x):
        out = {}
        x = self.fc(x) # (batch, time, 1)
        out['x_mix'] = x.clone()
        x = self.gaussian_smoothing(x.permute(0, 2, 1))
        out['x_smth'] = x.clone()
        x = self.rectification(x.squeeze(1))
        x = x / (x.max(dim=1, keepdim=True).values + self.eps)
        out['x_act'] = x
        return out

