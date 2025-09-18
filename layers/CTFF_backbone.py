from layers.FourierCorrelation import *
from layers.MultiWaveletCorrelation import *
from layers.FED_layers import *
from layers.TST_layers import *
from layers.RevIN import RevIN


# Cell
class CTFF_backbone(nn.Module):
    def __init__(self, configs, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int, max_seq_len: Optional[int] = 1024,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0., act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True, pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True, subtract_last=False, shared_embedding=True,
                 verbose: bool = False, **kwargs):

        super().__init__()

        self.configs = configs
        self.fusion_mode = getattr(configs, 'fusion_mode', 'context_gate')
        self.omega_fixed = float(getattr(configs, 'omega', 0.6))
        self.gate_tau = float(getattr(configs, 'gate_tau', 1.0))

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                    n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # FED_Encoder(FEB or FEA)
        self.FED_Encoder = Fourier_Model(configs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

        self.out_linear = nn.Linear(configs.dec_in, configs.dec_in)

        C = configs.dec_in
        if self.fusion_mode == 'context_gate':
            self.fuse_gate = nn.Sequential(
                nn.Conv1d(2 * C, C, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(C, C, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

            for m in self.fuse_gate.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        elif self.fusion_mode in ['scalar', 'vector']:
            shape = (1, 1, 1) if self.fusion_mode == 'scalar' else (1, C, 1)
            self.omega_raw = nn.Parameter(torch.zeros(*shape))
        self.last_gate_mean = None

    def forward(self, z, w: float = 1):  # z: [bs x nvars x seq_len]

        # revin
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        FED_encoder_data = z.permute(0, 2, 1)
        FED_encoder = self.FED_Encoder(FED_encoder_data)  # [B, pred_len, C]
        FED_encoder = FED_encoder.permute(0, 2, 1)        # -> [B, C, pred_len]

        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2)                                          # [B, C, patch_len, patch_num]

        if self.configs.FED_control:
            z = self.backbone(z)   # [B, C, d_model, patch_num]
            z = self.head(z)       # [B, C, pred_len]

            if self.fusion_mode == 'context_gate':
                # concat([time, freq]) along channel dim -> [B, 2C, T]
                gate_in = torch.cat([z, FED_encoder], dim=1)
                w_tensor = self.fuse_gate(gate_in)  # [B, C, T] in (0,1)
                if abs(self.gate_tau - 1.0) > 1e-8:
                    w_tensor = torch.sigmoid(torch.logit(w_tensor.clamp(1e-6, 1-1e-6)) / self.gate_tau)
                self.last_gate_mean = w_tensor.mean().detach()

            elif self.fusion_mode == 'fixed':
                w_tensor = torch.full_like(z, self.omega_fixed)
                self.last_gate_mean = w_tensor.mean().detach()

            elif self.fusion_mode in ['scalar', 'vector']:
                w_tensor = torch.sigmoid(self.omega_raw).expand_as(z)
                self.last_gate_mean = w_tensor.mean().detach()

            else:
                raise ValueError(f'Unsupported fusion_mode: {self.fusion_mode}')

            z = w_tensor * FED_encoder + (1.0 - w_tensor) * z
            z = self.out_linear(z.permute(0, 2, 1)).permute(0, 2, 1)

        else:
            z = self.backbone(z)   # [B, C, d_model, patch_num]
            z = self.head(z)       # [B, C, pred_len]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)

        return z


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)               # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)            # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]
        # 输入x维度:[batch,feature,patch_len,patch_num]
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)            # [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                      # [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [bs*nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)                  # [bs*nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)            # [bs x nvars x d_model x patch_num]

        return z


class Fourier_Model(nn.Module):  # FEDformer.py-Model

    def __init__(self, configs):
        super(Fourier_Model, self).__init__()
        self.configs = configs
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        if configs.Encorder == 'FEB':
            if configs.version == 'Wavelets':
                encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            else:
                encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                                out_channels=configs.d_model,
                                                seq_len=self.seq_len,
                                                modes=configs.modes,
                                                mode_select_method=configs.mode_select)
        else:
            if configs.version == 'Wavelets':
                encoder_self_att = MultiWaveletCross(in_channels=configs.d_model,
                                                     out_channels=configs.d_model,
                                                     seq_len_q=self.seq_len // 2 + self.pred_len,
                                                     seq_len_kv=self.seq_len,
                                                     modes=configs.modes,
                                                     ich=configs.d_model,
                                                     base=configs.base,
                                                     activation=configs.cross_activation)
            else:
                encoder_self_att = FourierCrossAttention(in_channels=configs.d_model,
                                                         out_channels=configs.d_model,
                                                         seq_len_q=self.seq_len//2+self.pred_len,
                                                         seq_len_kv=self.seq_len,
                                                         modes=configs.modes,
                                                         mode_select_method=configs.mode_select)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.FED_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        self.lstm = nn.LSTM(configs.d_model, configs.d_model, 1)
        self.linear = nn.Linear(configs.d_model, configs.dec_in)

    def forward(self, x_enc, enc_self_mask=None):
        # enc 编码
        enc_out = self.enc_embedding(x_enc)
        enc_out, enc_attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out, (h_n, c_n) = self.lstm(enc_out)
        enc_out = self.linear(enc_out[:, -self.configs.pred_len:, :])  # [B, pred_len, dec_in]
        return enc_out
