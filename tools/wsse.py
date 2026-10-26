import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmseg.registry import MODELS

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_scan
except Exception:
    _mamba_scan = None

def haar_dwt2d(x):

    h, w = x.shape[-2:]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh

def haar_idwt2d(ll, lh, hl, hh, out_size=None):

    b, c, h, w = ll.shape
    out = ll.new_empty(b, c, h * 2, w * 2)
    out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    out[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    out[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    if out_size is not None and tuple(out.shape[-2:]) != tuple(out_size):
        out = out[:, :, :out_size[0], :out_size[1]]
    return out

_MAX_EXP = 40.0

def _selective_scan_torch(u, delta, A, B, C, D=None, chunk=64):

    b, d, L = u.shape
    n = A.shape[1]

    pad = (-L) % chunk
    Lp = L + pad
    if pad:
        u = F.pad(u, (0, pad))
        delta = F.pad(delta, (0, pad))
        B = F.pad(B, (0, pad))
        C = F.pad(C, (0, pad))

    nc = Lp // chunk
    u = u.view(b, d, nc, chunk)
    delta = delta.view(b, d, nc, chunk)
    Bv = B.view(b, n, nc, chunk)
    Cv = C.view(b, n, nc, chunk)

    dA = (delta.unsqueeze(-1) * A.view(1, d, 1, 1, n)).permute(0, 1, 2, 4, 3).contiguous()

    cum = torch.cumsum(dA, dim=-1)

    dBu = (delta.unsqueeze(3)
           * Bv.permute(0, 2, 1, 3).unsqueeze(1)
           * u.unsqueeze(3))
    decay = torch.exp(torch.clamp(-cum, max=_MAX_EXP))
    h_local = torch.exp(cum) * torch.cumsum(dBu * decay, dim=-1)
    Cp = Cv.permute(0, 2, 1, 3).contiguous()
    y_local = (h_local * Cp.unsqueeze(1)).sum(dim=3)

    chunk_decay = torch.exp(cum[..., -1])
    h_end = h_local[..., -1]

    ys = []
    prev = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    for ci in range(nc):

        carry = (torch.exp(cum[:, :, ci]) * prev.unsqueeze(-1)
                 * Cp[:, ci].unsqueeze(1)).sum(dim=2)
        ys.append(y_local[:, :, ci] + carry)
        prev = chunk_decay[:, :, ci] * prev + h_end[:, :, ci]

    y = torch.cat(ys, dim=-1)
    if D is not None:
        y = y + u.reshape(b, d, Lp) * D.view(1, d, 1)
    return y[..., :L]

def selective_scan(u, delta, A, B, C, D=None, chunk=64, use_mamba=False):

    if use_mamba and _mamba_scan is not None and u.is_cuda:
        return _mamba_scan(u, delta, A, B, C, D=D, delta_softplus=False)
    return _selective_scan_torch(u, delta, A, B, C, D=D, chunk=chunk)

class SelectiveStateSpace2D(BaseModule):

    _VALID_SCAN_MODES = ('raster', 'snake_h', 'snake_v', 'cross')

    def __init__(self,
                 dim,
                 d_state=4,
                 dt_rank=None,
                 dt_limit=(1e-4, 1e-1),
                 bidirectional=True,
                 chunk=64,
                 use_mamba=False,
                 scan_mode='raster',
                 energy_mod=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if scan_mode not in self._VALID_SCAN_MODES:
            raise ValueError(
                f'SelectiveStateSpace2D: scan_mode must be one of '
                f'{self._VALID_SCAN_MODES}, got {scan_mode!r}')
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, dim // 16)
        self.dt_limit = dt_limit
        self.bidirectional = bidirectional
        self.chunk = chunk
        self.use_mamba = use_mamba
        self.scan_mode = scan_mode
        self.energy_mod = energy_mod

        self.x_proj = nn.Linear(dim, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, dim, bias=True)

        a_log = torch.log(torch.arange(1, d_state + 1).float())
        self.A_log = nn.Parameter(a_log[None].repeat(dim, 1))
        self.D = nn.Parameter(torch.ones(dim))

        dt = torch.exp(torch.rand(dim) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        if energy_mod:
            self.energy_gain = nn.Parameter(torch.zeros(1))

    def _scan_once(self, u, delta, B, C, A):
        return selective_scan(u, delta, A, B, C, D=self.D,
                              chunk=self.chunk, use_mamba=self.use_mamba)

    def _scan_seq(self, seq, energy=None):

        z = self.x_proj(seq)
        dt_r, bp, cp = torch.split(z, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_r)).clamp(*self.dt_limit)

        if self.energy_mod and energy is not None:

            gate = 1.0 + torch.tanh(self.energy_gain) * (0.5 - energy)
            delta = (delta * gate).clamp(*self.dt_limit)

        u = seq.transpose(1, 2)
        dt = delta.transpose(1, 2)
        B = bp.transpose(1, 2)
        Cc = cp.transpose(1, 2)
        A = -torch.exp(self.A_log)

        y = self._scan_once(u, dt, B, Cc, A)
        if self.bidirectional:
            uf = torch.flip(u, dims=[-1])
            y = y + torch.flip(
                self._scan_once(uf, torch.flip(dt, dims=[-1]),
                                torch.flip(B, dims=[-1]), torch.flip(Cc, dims=[-1]), A),
                dims=[-1])
        return y

    @staticmethod
    def _snake_order(h, w, device):

        idx = torch.arange(h * w, device=device).view(h, w)
        idx[1::2] = idx[1::2].flip(-1)
        flat = idx.reshape(-1)
        inv = torch.empty_like(flat)
        inv[flat] = torch.arange(h * w, device=device)
        return flat, inv

    def _scan_directional(self, x, vertical, energy=None):

        b, c, h, w = x.shape
        src = x.transpose(-2, -1) if vertical else x
        hh, ww = src.shape[-2:]
        flat, inv = self._snake_order(hh, ww, x.device)
        seq = src.flatten(2)[:, :, flat].transpose(1, 2)
        e = None
        if energy is not None:
            esrc = energy.transpose(-2, -1) if vertical else energy
            e = esrc.flatten(2)[:, :, flat].transpose(1, 2)
        y = self._scan_seq(seq, e)
        y = y[:, :, inv].view(b, c, hh, ww)
        return y.transpose(-2, -1) if vertical else y

    def forward(self, x, scan_mode=None, energy=None):

        b, c, h, w = x.shape
        mode = self.scan_mode if scan_mode is None else scan_mode

        if mode == 'raster':

            seq = x.flatten(2).transpose(1, 2)
            e = None if energy is None else energy.flatten(2).transpose(1, 2)
            return self._scan_seq(seq, e).view(b, c, h, w)

        if mode == 'cross':

            y = self._scan_directional(x, vertical=False, energy=energy)
            y = y + self._scan_directional(x, vertical=True, energy=energy)
            return y.view(b, c, h, w)

        y = self._scan_directional(x, vertical=(mode == 'snake_v'),
                                   energy=energy)
        return y.view(b, c, h, w)

class AnisoDWConv(BaseModule):

    def __init__(self, channels, kernel_len=5,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if channels % 4 != 0:
            raise ValueError(
                f'AnisoDWConv: channels 必须能被 4 整除（要沿通道切 4 组），'
                f'当前 channels={channels}')
        if kernel_len % 2 == 0 or kernel_len <= 1:
            raise ValueError(
                f'AnisoDWConv: kernel_len 必须是 >1 的奇数，当前 {kernel_len}')
        g = channels // 4
        pad = kernel_len // 2
        self.groups = g
        self.kernel_len = kernel_len

        self.dw_sq = nn.Conv2d(g, g, kernel_size=3, padding=1,
                               groups=g, bias=False)
        self.dw_h = nn.Conv2d(g, g, kernel_size=(1, kernel_len),
                              padding=(0, pad), groups=g, bias=False)
        self.dw_v = nn.Conv2d(g, g, kernel_size=(kernel_len, 1),
                              padding=(pad, 0), groups=g, bias=False)
        self.dw_dil = nn.Conv2d(g, g, kernel_size=3, padding=2, dilation=2,
                                groups=g, bias=False)

        self.fuse = ConvModule(channels, channels, kernel_size=1,
                               norm_cfg=norm_cfg, act_cfg=act_cfg,
                               inplace=False)

    def forward(self, x):
        c0, c1, c2, c3 = torch.chunk(x, 4, dim=1)
        y = torch.cat([self.dw_sq(c0), self.dw_h(c1),
                       self.dw_v(c2), self.dw_dil(c3)], dim=1)
        return self.fuse(y)

def _build_hf_dw(channels, norm_cfg, act_cfg, aniso_dw=False, aniso_kernel=5):

    if aniso_dw:
        return AnisoDWConv(channels, kernel_len=aniso_kernel,
                           norm_cfg=norm_cfg, act_cfg=act_cfg)
    return ConvModule(channels, channels, kernel_size=3, padding=1,
                      groups=channels, norm_cfg=norm_cfg, act_cfg=act_cfg,
                      inplace=False)

class WaveletSSMBlock(BaseModule):

    _VALID_STREAMS = ('both', 'low_only', 'high_only')
    _VALID_FUSIONS = ('sum', 'scaled', 'gate', 'csaf')
    _VALID_CSAF_MODES = ('parallel', 'serial')
    _VALID_SCAN_MODES = ('raster', 'directional', 'subband')
    _VALID_BAND_SELECTS = ('soft', 'sparse')

    def __init__(self,
                 channels,
                 ssm_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 band_gate=True,
                 use_ssm=True,
                 streams='both',
                 fusion='sum',
                 csaf_mode='parallel',
                 scan_mode='raster',
                 band_select='soft',
                 band_topk=None,
                 energy_mod=False,
                 aniso_dw=False,
                 aniso_kernel=5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if streams not in self._VALID_STREAMS:
            raise ValueError(
                f'WaveletSSMBlock: streams must be one of '
                f'{self._VALID_STREAMS}, got {streams!r}')
        if fusion not in self._VALID_FUSIONS:
            raise ValueError(
                f'WaveletSSMBlock: fusion must be one of '
                f'{self._VALID_FUSIONS}, got {fusion!r}')
        if csaf_mode not in self._VALID_CSAF_MODES:
            raise ValueError(
                f'WaveletSSMBlock: csaf_mode must be one of '
                f'{self._VALID_CSAF_MODES}, got {csaf_mode!r}')
        if scan_mode not in self._VALID_SCAN_MODES:
            raise ValueError(
                f'WaveletSSMBlock: scan_mode must be one of '
                f'{self._VALID_SCAN_MODES}, got {scan_mode!r}')
        if energy_mod and (scan_mode != 'subband' or not use_ssm):
            raise ValueError(
                f'WaveletSSMBlock: energy_mod=True 需要 scan_mode="subband" 且 '
                f'use_ssm=True（能量调制作用在**子带各自的** SSM 状态步长 Δ 上，'
                f'因此必须先有「子带独立扫描」这个前提），'
                f'当前 scan_mode={scan_mode!r} use_ssm={use_ssm!r}')
        if band_select not in self._VALID_BAND_SELECTS:
            raise ValueError(
                f'WaveletSSMBlock: band_select must be one of '
                f'{self._VALID_BAND_SELECTS}, got {band_select!r}')
        if band_select != 'soft' and not band_gate:
            raise ValueError(
                f"WaveletSSMBlock: band_select={band_select!r} 需要 band_gate=True"
                f'（没有可学习的频带权重就无从稀疏化），'
                f'当前 band_gate={band_gate!r}')
        if fusion != 'sum' and streams != 'both':
            raise ValueError(
                f"WaveletSSMBlock: fusion={fusion!r} 只在 streams='both' 时才有"
                f'意义（单流没有「两路」可融合），当前 streams={streams!r}')
        self.band_gate = band_gate
        self.use_ssm = use_ssm
        self.streams = streams
        self.fusion = fusion
        self.csaf_mode = csaf_mode
        self.scan_mode = scan_mode
        self.band_select = band_select
        self.band_topk = band_topk
        self.energy_mod = energy_mod
        if aniso_dw and channels % 4 != 0:
            raise ValueError(
                f'WaveletSSMBlock: aniso_dw=True 需要 channels 能被 4 整除'
                f'（AnisoDWConv 要沿通道切 4 组），当前 channels={channels}')
        if aniso_dw and streams == 'low_only':
            raise ValueError(
                f"WaveletSSMBlock: aniso_dw=True 替换的是**高频支**的 hf_dw，"
                f"但 streams='low_only' 根本不构建高频支 —— 该开关会被静默忽略。"
                f"请改用 streams='both' 或 'high_only'。")
        self.aniso_dw = aniso_dw
        self.aniso_kernel = aniso_kernel

        zero_init = []

        if streams != 'high_only':
            self.low_delta = nn.Conv2d(channels, channels, kernel_size=3,
                                       padding=1, groups=channels)
            zero_init.append(self.low_delta)

        if streams != 'low_only':
            if band_gate:

                self.band_logits = nn.Parameter(torch.zeros(3, channels))

            if scan_mode == 'subband':

                self.hf_proj = ConvModule(
                    channels, channels, kernel_size=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
                self.hf_dw = _build_hf_dw(
                    channels, norm_cfg, act_cfg,
                    aniso_dw=aniso_dw, aniso_kernel=aniso_kernel)
                if use_ssm:

                    _ssm_cfg = dict(ssm_cfg or {})
                    _ssm_cfg['scan_mode'] = 'raster'
                    _ssm_cfg['energy_mod'] = energy_mod
                    self.ssm = SelectiveStateSpace2D(channels, **_ssm_cfg)

                self.hf_delta = nn.Conv2d(channels, channels, kernel_size=1)
                zero_init.append(self.hf_delta)
            else:
                self.hf_proj = ConvModule(
                    3 * channels, channels, kernel_size=1,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False)
                self.hf_dw = _build_hf_dw(
                    channels, norm_cfg, act_cfg,
                    aniso_dw=aniso_dw, aniso_kernel=aniso_kernel)
                if use_ssm:

                    _ssm_cfg = dict(ssm_cfg or {})
                    _ssm_cfg['scan_mode'] = (
                        'cross' if scan_mode == 'directional' else 'raster')
                    self.ssm = SelectiveStateSpace2D(channels, **_ssm_cfg)

                self.hf_delta = nn.Conv2d(channels, 3 * channels, kernel_size=1)
                zero_init.append(self.hf_delta)

        if streams == 'both' and fusion == 'scaled':

            self.res_scale = nn.Parameter(torch.ones(2))
        elif streams == 'both' and fusion == 'gate':

            self.gate_proj = nn.Conv2d(channels, channels, kernel_size=1)
            zero_init.append(self.gate_proj)
        elif streams == 'both' and fusion == 'csaf':

            hidden = max(4, channels // 8)
            self.csaf_ch = nn.Sequential(
                nn.Linear(4 * channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 2 * channels),
            )
            zero_init.append(self.csaf_ch[-1])

            self.csaf_sp = nn.Conv2d(2, 1, kernel_size=7, padding=3)
            zero_init.append(self.csaf_sp)

        for m in zero_init:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _subband_branch(self, band, scan):

        f = self.hf_dw(self.hf_proj(band))
        if self.use_ssm:
            e = None
            if self.energy_mod:

                e = band.abs().mean(dim=1, keepdim=True)
                e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-6)
            f = self.ssm(f, scan_mode=scan, energy=e)
        return self.hf_delta(f)

    def _band_weights(self):

        logits = self.band_logits
        if self.band_select == 'soft':
            a = torch.softmax(logits, dim=0)
        else:
            a = 1.0 + torch.tanh(logits)
            k = self.band_topk
            if k is not None and k < a.shape[0]:

                order = a.abs().argsort(dim=0, descending=True)
                rank = torch.empty_like(order)
                rank.scatter_(0, order,
                              torch.arange(a.shape[0], device=a.device)
                              .view(-1, 1).expand_as(order))
                keep = (rank < k).to(a.dtype)
                a_hard = a * keep
                a = a + (a_hard - a).detach()
        return [a[i].view(1, -1, 1, 1) for i in range(a.shape[0])]

    def _fusion_weights(self, ll):

        if self.streams != 'both' or self.fusion == 'sum':
            return None, None
        if self.fusion == 'scaled':
            return self.res_scale[0], self.res_scale[1]

        t = torch.tanh(self.gate_proj(ll))
        return 1.0 - t, 1.0 + t

    def _csaf_channel(self, f):

        desc = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return 1.0 + torch.tanh(self.csaf_ch(desc))[:, :, None, None]

    def _csaf_spatial(self, f):

        sp = torch.cat([f.mean(dim=1, keepdim=True),
                        f.amax(dim=1, keepdim=True)], dim=1)
        return 1.0 + torch.tanh(self.csaf_sp(sp))

    def _forward_csaf(self, x, out_size, d_low, d_lh, d_hl, d_hh):

        z = torch.zeros_like(d_low)
        r_low = haar_idwt2d(d_low, z, z, z, out_size=out_size)
        r_high = haar_idwt2d(z, d_lh, d_hl, d_hh, out_size=out_size)
        joint = torch.cat([r_low, r_high], dim=1)

        if self.csaf_mode == 'serial':

            joint = joint * self._csaf_channel(joint)
            joint = joint * self._csaf_spatial(joint)
        else:

            joint = joint * self._csaf_channel(joint) * self._csaf_spatial(joint)

        r_low, r_high = joint.chunk(2, dim=1)
        return x + r_low + r_high

    def forward(self, x):

        out_size = x.shape[-2:]

        ll, lh, hl, hh = haar_dwt2d(x)

        d_lh = d_hl = d_hh = None
        if self.streams != 'low_only':
            bands = [lh, hl, hh]
            if self.band_gate:
                bands = [b * w for b, w in zip(bands, self._band_weights())]

            if self.scan_mode == 'subband':

                d_lh = self._subband_branch(bands[0], 'snake_h')
                d_hl = self._subband_branch(bands[1], 'snake_v')
                d_hh = self._subband_branch(bands[2], 'raster')
            else:
                hf = self.hf_dw(self.hf_proj(torch.cat(bands, dim=1)))
                if self.use_ssm:
                    hf = self.ssm(hf)
                d_lh, d_hl, d_hh = torch.chunk(self.hf_delta(hf), 3, dim=1)

        d_low = None if self.streams == 'high_only' else self.low_delta(ll)

        if self.streams == 'both' and self.fusion == 'csaf':
            return self._forward_csaf(x, out_size, d_low, d_lh, d_hl, d_hh)

        w_low, w_high = self._fusion_weights(ll)

        if w_low is None:

            ll_e = ll if self.streams == 'high_only' else ll + d_low
            if d_lh is None:
                return haar_idwt2d(ll_e, lh, hl, hh, out_size=out_size)
            return haar_idwt2d(ll_e, lh + d_lh, hl + d_hl, hh + d_hh,
                               out_size=out_size)

        ll_e = ll if self.streams == 'high_only' else ll + w_low * d_low
        if d_lh is None:
            return haar_idwt2d(ll_e, lh, hl, hh, out_size=out_size)
        return haar_idwt2d(ll_e, lh + w_high * d_lh, hl + w_high * d_hl,
                           hh + w_high * d_hh, out_size=out_size)

@MODELS.register_module()
class WSSE(BaseModule):

    def __init__(self,
                 in_channels,
                 levels=(0, 1),
                 ssm_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 band_gate=True,
                 use_ssm=True,
                 streams='both',
                 fusion='sum',
                 csaf_mode='parallel',
                 scan_mode='raster',
                 band_select='soft',
                 band_topk=None,
                 energy_mod=False,
                 aniso_dw=False,
                 aniso_kernel=5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = list(in_channels)
        self.levels = tuple(levels)

        for lv in self.levels:
            if not 0 <= lv < len(self.in_channels):
                raise ValueError(
                    f'WSSE: level {lv} out of range for in_channels={in_channels}')

        self.blocks = nn.ModuleList([
            WaveletSSMBlock(self.in_channels[lv], ssm_cfg=ssm_cfg,
                            norm_cfg=norm_cfg, act_cfg=act_cfg,
                            band_gate=band_gate, use_ssm=use_ssm,
                            streams=streams, fusion=fusion,
                            csaf_mode=csaf_mode, scan_mode=scan_mode,
                            band_select=band_select, band_topk=band_topk,
                            energy_mod=energy_mod,
                            aniso_dw=aniso_dw, aniso_kernel=aniso_kernel)
            for lv in self.levels
        ])

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels), (
            f'WSSE expected {len(self.in_channels)} levels, got {len(inputs)}')
        outs = list(inputs)
        for k, lv in enumerate(self.levels):
            outs[lv] = self.blocks[k](outs[lv])
        return tuple(outs)
