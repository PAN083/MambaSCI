from __future__ import annotations
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
# from mamba_ssm import Mamba2Simple
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode,InterpolateMode
from timm.models.layers import DropPath,trunc_normal_
from monai.networks.blocks.upsample import UpSample
from colour.hints import NDArray, Tuple
from colour.utilities import as_float_array, tstack
from cacti.utils.utils import A, At
from typing import Tuple


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, nf, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, nf, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_upsample_layer(spatial_dims: int, in_channels: int, upsample_mode: UpsampleMode | str = "nontrainable", scale_factor: int = 2):
    if spatial_dims == 2:
        scale_factor = (scale_factor, scale_factor)
    elif spatial_dims == 3:
        scale_factor = (1, scale_factor, scale_factor)
    else:
        raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")

    return UpSample(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        scale_factor=scale_factor,
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=8):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            # 特征图缩放到0，1之间，表示通道注意力权重
            nn.Sigmoid())

    def forward(self, x):
        # y 为通道注意力权重
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4,squeeze_factor=8):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            #卷积核为3 步长1 填充1
            nn.Conv3d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

    
def get_dwconv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels, 
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, 
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, nf, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x



class ParallelChunkProcessor(nn.Module):
    def __init__(self, original_forward_func):
        super().__init__()
        self.original_forward_func = original_forward_func

    def forward(self, x_chunk):
        return self.original_forward_func(x_chunk)


class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2, mlp_ratio=4,drop_path=0.,drop=0.,act_layer=nn.GELU):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm1 = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nframes=8
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale1= nn.Parameter(torch.ones(1))
        self.skip_scale2= nn.Parameter(torch.ones(1))
        self.skip_scale3= nn.Parameter(torch.ones(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(output_dim)
        self.conv_blk = CAB(output_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(in_features=input_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        
        B, C, nf, H, W= x.shape
        assert C == self.input_dim

        num_chunk = nf // 8
        chunks = [x[:, :, i*8:(i+1)*8, :, :] for i in range(num_chunk)]

    # 创建并行处理器
        parallel_processor = ParallelChunkProcessor(self._process_chunk)
        
        # 使用parallel_apply并行处理所有chunk
        outputs = nn.parallel.parallel_apply([parallel_processor] * num_chunk, chunks)

        # 在时间维度上拼接结果
        final_out = torch.cat(outputs, dim=2)
        return final_out
    
    def _process_chunk(self, x_chunk):
        # outputs = []
        # for i in range(num_chunk):
        #     x_chunk = x[:, :, i*8:(i+1)*8, :, :]
        

        # n_tokens = 131072
        B,C,T,H,W = x_chunk.shape
        n_tokens = x_chunk.shape[2:].numel()
        img_dims = x_chunk.shape[2:]
        x_flat = x_chunk.reshape(B, C, n_tokens).transpose(-1, -2)
    
    # x_norm = self.norm1(x_flat)
    # x1, x2, x3, x4 = torch.chunk(x_norm,4,dim=2)
    # x_mamba1 = self.drop_path(self.mamba(x1))
    # x_mamba2 = self.drop_path(self.mamba(x2))
    # x_mamba3 = self.drop_path(self.mamba(x3))
    # x_mamba4 = self.drop_path(self.mamba(x4))
    # x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)
    # x_mamba = x_flat*self.skip_scale1 + x_mamba

        '''Residual connection'''
        x_mamba = x_flat * self.skip_scale1 + self.drop_path(self.mamba(self.norm1(x_flat)))
        x_mamba = x_mamba * self.skip_scale2 + self.drop_path(self.mlp(self.norm2(x_mamba), 8, H, W))
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        # B C T H W
        out = out.permute(0,2,3,4,1).contiguous()
        out = out * self.skip_scale3 + self.conv_blk(self.norm3(out).permute(0,4,1,2,3).contiguous()).permute(0,2,3,4,1).contiguous()
        out = out.permute(0,4,1,2,3).contiguous()
        return out

    #     outputs.append(out)
    # final_out = torch.cat(outputs, dim=2)
    # return final_out
        '''no learnable scale'''
        # x_mamba = x_flat  + self.drop_path(self.mamba(self.norm1(x_flat)))
        # x_mamba = x_mamba  + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        # x_mamba = self.proj(x_mamba)
        # out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        # #B C T H W
        # out = out.permute(0,2,3,4,1).contiguous()
        # out = out  + self.conv_blk(self.norm3(out).permute(0,4,1,2,3).contiguous()).permute(0,2,3,4,1).contiguous()
        # out = out.permute(0,4,1,2,3).contiguous()
        # return out

        # x_mamba = self.drop_path(self.mamba(self.norm1(x_flat)))
        # x_mamba = self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        # x_mamba = self.proj(x_mamba)
        # out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        # #B C T H W
        # out = out.permute(0,2,3,4,1).contiguous()
        # out = self.conv_blk(self.norm3(out).permute(0,4,1,2,3).contiguous()).permute(0,2,3,4,1).contiguous()
        # out = out.permute(0,4,1,2,3).contiguous()
        # return out

def get_mamba_layer(
    spatial_dims: int, in_channels: int, out_channels: int, nframes: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims==2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        
        if spatial_dims==3:
            return nn.Sequential(mamba_layer,nn.MaxPool3d(kernel_size=(1, stride, stride),
                    stride=(1, stride, stride))
)
            # return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


class ResMambaBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        n_frames: int, 
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, nframes=n_frames
        )
        self.conv2 = get_mamba_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels,nframes=n_frames
        )

    def forward(self, x):
        # print(x.shape)
        identity = x

        x = self.norm1(x)
        # print(x.shape)
        x = self.act(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.norm2(x)
        # print(x.shape)
        x = self.act(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)

        x += identity
        # print(x.shape)
        return x


class ResUpBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.skip_scale= nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x


class MambaSCI(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 16,
        in_channels: int = 1,
        out_channels: int = 1,
        n_frames: int= 8,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (2,4,4,6),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        self.out_channels = out_channels
        self.n_frames = n_frames
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)
        
        # self.net = BasicVSRPlusPlus()

#         self.conv2 = nn.Sequential(

#         nn.Conv3d(init_filters, init_filters*2, kernel_size=3, stride=1, padding=1),
#         nn.LeakyReLU(inplace=True),
#         nn.Conv3d(init_filters*2, init_filters, kernel_size=3, stride=1,padding=1),
#         nn.LeakyReLU(inplace=True),
#         nn.Conv3d(init_filters, 3, kernel_size=1, stride=1,padding=0),
# )

        self.conv2 = nn.Sequential(
        nn.Conv3d(init_filters, init_filters//2, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(init_filters//2, init_filters//4, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(init_filters//4, 3, kernel_size=1, stride=1),
)


        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            downsample_mamba = (
                get_mamba_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, 8, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba, *[ResMambaBlock(spatial_dims, layer_in_channels, 8 ,norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResUpBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_dwconv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # print(x.shape)
        x = self.convInit(x)

        down_x = []
        down_x.append(x)

        if self.dropout_prob is not None:
            x = self.dropout(x)
        # down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)
        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        # c_list = [self.init_filters*4 , self.init_filters*2,self.init_filters]
        # self.scab = SC_Att_Bridge(c_list)
        # # self.scab = SC_Att_Bridge()
        # t1, t2, t3 = self.scab(down_x[1],down_x[2],down_x[3])
        # for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
        #     if i == 0:
        #         x = up(x) + t1
        #         x = upl(x)
        #     elif i == 1:
        #         x = up(x) + t2
        #         x = upl(x)
        #     elif i == 2:
        #         x = up(x) + t3
                # x = upl(x)
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
        if self.use_conv_final:
    
            x  = x + down_x[-1]
            x = self.conv2(x)
            # x = self.conv_final(x)
            
            # x = self.net(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:

        x, down_x = self.encode(x)
        down_x.reverse()
        x = self.decode(x, down_x)
        return x
    
    def forward(self,y,Phi,Phi_s):
        out_list = []
        x = At(y,Phi)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.forward_features(x)
        
        out_list.append(x)
        return out_list 





# from __future__ import annotations
# import math

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # from mamba_ssm import Mamba
# from mamba_ssm import Mamba2Simple
# from monai.networks.blocks.convolutions import Convolution
# from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer
# from monai.networks.layers.factories import Dropout
# from monai.networks.layers.utils import get_act_layer, get_norm_layer
# from monai.utils import UpsampleMode,InterpolateMode
# from timm.models.layers import DropPath,trunc_normal_
# from monai.networks.blocks.upsample import UpSample
# from colour.hints import NDArray, Tuple
# from colour.utilities import as_float_array, tstack
# import cacti.models.model.arch_util as arch_util

# from mmcv.runner import load_checkpoint
# from mmedit.utils import get_root_logger

# # from mmedit.models.backbones.sr_backbones.basicvsr_pp import BasicVSRPlusPlus
# from cacti.utils.utils import A, At
# from typing import Tuple
# from wtconv.wtconv2d import WTConv2d


# def get_upsample_layer(spatial_dims: int, in_channels: int, upsample_mode: UpsampleMode | str = "nontrainable", scale_factor: int = 2):
#     if spatial_dims == 2:
#         scale_factor = (scale_factor, scale_factor)
#     elif spatial_dims == 3:
#         scale_factor = (1, scale_factor, scale_factor)
#     else:
#         raise ValueError(f"Unsupported spatial_dims: {spatial_dims}")

#     return UpSample(
#         spatial_dims=spatial_dims,
#         in_channels=in_channels,
#         out_channels=in_channels,
#         scale_factor=scale_factor,
#         mode=upsample_mode,
#         interp_mode=InterpolateMode.LINEAR,
#         align_corners=False,
#     )

# class ChannelAttention(nn.Module):
#     """Channel attention used in RCAN.
#     Args:
#         num_feat (int): Channel number of intermediate features.
#         squeeze_factor (int): Channel squeeze factor. Default: 16.
#     """

#     def __init__(self, num_feat, squeeze_factor=8):
#         super(ChannelAttention, self).__init__()
#         self.attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
#             # 特征图缩放到0，1之间，表示通道注意力权重
#             nn.Sigmoid())

#     def forward(self, x):
#         # y 为通道注意力权重
#         y = self.attention(x)
#         return x * y


# class CAB(nn.Module):
#     def __init__(self, num_feat, compress_ratio=4,squeeze_factor=8):
#         super(CAB, self).__init__()
#         self.cab = nn.Sequential(
#             #卷积核为3 步长1 填充1
#             nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
#             nn.GELU(),
#             nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
#             ChannelAttention(num_feat, squeeze_factor)
#         )

#     def forward(self, x):
#         return self.cab(x)

    
# def get_dwconv_layer(
#     spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
# ):
#     depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels, 
#                              strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
#     point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, 
#                              strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
#     return torch.nn.Sequential(depth_conv, point_conv)

# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x, nf, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, nf, H, W)
#         x = self.dwconv(x)
#         x = x.flatten(2).transpose(1, 2)

#         return x

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.dwconv = DWConv(hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, nf, H, W):
#         x = self.fc1(x)
#         x = self.dwconv(x, nf, H, W)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x




# class MambaLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, nframes, d_state = 16, d_conv = 4, expand = 2, mlp_ratio=4,drop_path=0.,drop=0.,headdim=4,act_layer=nn.GELU):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.norm1 = nn.LayerNorm(input_dim)
#         self.mamba = Mamba2Simple(
#                 d_model=input_dim, # Model dimension d_model
#                 d_state=d_state,  # SSM state expansion factor
#                 d_conv=d_conv,    # Local convolution width
#                 expand=expand,    # Block expansion factor
#                 headdim=headdim,
#                 nframes=nframes
                
#         )
#         self.proj = nn.Linear(input_dim, output_dim)
#         self.skip_scale1= nn.Parameter(torch.ones(1))
#         self.skip_scale2= nn.Parameter(torch.ones(1))
#         self.skip_scale3= nn.Parameter(torch.ones(1))
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = nn.LayerNorm(input_dim)
#         self.norm3 = nn.LayerNorm(output_dim)
#         self.conv_blk = CAB(output_dim)
#         mlp_hidden_dim = int(input_dim * mlp_ratio)
#         self.mlp = Mlp(in_features=input_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
#     def forward(self, x):
#         if x.dtype == torch.float16:
#             x = x.type(torch.float32)
        
#         B, C, nf, H, W= x.shape
#         assert C == self.input_dim
#         # n_tokens = 131072
#         n_tokens = x.shape[2:].numel()
#         img_dims = x.shape[2:]
#         x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        
#         # x_norm = self.norm1(x_flat)
#         # x1, x2, x3, x4 = torch.chunk(x_norm,4,dim=2)
#         # x_mamba1 = self.drop_path(self.mamba(x1))
#         # x_mamba2 = self.drop_path(self.mamba(x2))
#         # x_mamba3 = self.drop_path(self.mamba(x3))
#         # x_mamba4 = self.drop_path(self.mamba(x4))
#         # x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)
#         # x_mamba = x_flat*self.skip_scale1 + x_mamba

#         '''Residual connection'''
#         x_mamba = x_flat * self.skip_scale1 + self.drop_path(self.mamba(self.norm1(x_flat)))
#         x_mamba = x_mamba * self.skip_scale2 + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
#         x_mamba = self.proj(x_mamba)
#         out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
#         # B C T H W
#         out = out.permute(0,2,3,4,1).contiguous()
#         # out = out * self.skip_scale3 + self.conv_blk(self.norm3(out).permute(0,4,1,2,3).contiguous()).permute(0,2,3,4,1).contiguous()
#         out = out * self.skip_scale3 + self.conv_blk(self.norm3(out).permute(0,1,4,2,3).contiguous().view(B*nf,-1,H,W)).view(B,nf,-1,H,W).permute(0,1,3,4,2).contiguous()
#         out = out.permute(0,4,1,2,3).contiguous()

#         '''no learnable scale'''
#         return out

# def get_mamba_layer(
#     spatial_dims: int, in_channels: int, out_channels: int, nframes: int, stride: int = 1
# ):
#     mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels, nframes= nframes)
#     if stride != 1:
#         if spatial_dims==2:
#             return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        
#         if spatial_dims==3:
#             return nn.Sequential(mamba_layer,nn.MaxPool3d(kernel_size=(1, stride, stride),
#                     stride=(1, stride, stride))
# )
#             # return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
#     return mamba_layer


# class ResMambaBlock(nn.Module):

#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         n_frames: int, 
#         norm: tuple | str,
#         kernel_size: int = 3,
#         act: tuple | str = ("RELU", {"inplace": True}),
#     ) -> None:
#         """
#         Args:
#             spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
#             in_channels: number of input channels.
#             norm: feature normalization type and arguments.
#             kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
#             act: activation type and arguments. Defaults to ``RELU``.
#         """

#         super().__init__()

#         if kernel_size % 2 != 1:
#             raise AssertionError("kernel_size should be an odd number.")

#         self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
#         self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
#         self.act = get_act_layer(act)
#         self.conv1 = get_mamba_layer(
#             spatial_dims, in_channels=in_channels, out_channels=in_channels, nframes=n_frames
#         )
#         self.conv2 = get_mamba_layer(
#             spatial_dims, in_channels=in_channels, out_channels=in_channels,nframes=n_frames
#         )

#     def forward(self, x):
#         # print(x.shape)
#         identity = x

#         x = self.norm1(x)
#         # print(x.shape)
#         x = self.act(x)
#         # print(x.shape)
#         x = self.conv1(x)
#         # print(x.shape)
#         x = self.norm2(x)
#         # print(x.shape)
#         x = self.act(x)
#         # print(x.shape)
#         x = self.conv2(x)
#         # print(x.shape)

#         x += identity
#         # print(x.shape)
#         return x


# class ResUpBlock(nn.Module):

#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         norm: tuple | str,
#         kernel_size: int = 3,
#         act: tuple | str = ("RELU", {"inplace": True}),
#     ) -> None:
#         """
#         Args:
#             spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
#             in_channels: number of input channels.
#             norm: feature normalization type and arguments.
#             kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
#             act: activation type and arguments. Defaults to ``RELU``.
#         """

#         super().__init__()

#         if kernel_size % 2 != 1:
#             raise AssertionError("kernel_size should be an odd number.")

#         self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
#         self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
#         self.act = get_act_layer(act)
#         self.conv = get_dwconv_layer(
#             spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
#         )
#         self.skip_scale= nn.Parameter(torch.ones(1))

#     def forward(self, x):
#         identity = x

#         x = self.norm1(x)
#         x = self.act(x)
#         x = self.conv(x) + self.skip_scale * identity
#         x = self.norm2(x)
#         x = self.act(x)
#         return x
    
# class ConvBlock(nn.Module):

#     def __init__(self, dim,
#                  drop_path=0.,
#                  layer_scale=None,
#                  kernel_size=3):
#         super().__init__()

#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
#         # self.conv1 = nn.Sequential(
#         #     WTConv2d(dim,dim,kernel_size=kernel_size,stride=1,wt_levels=3),
#         #     nn.Conv2d(dim,dim,kernel_size=1)
#         # )
#         self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
#         self.act1 = nn.GELU(approximate= 'tanh')
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
#         # self.conv2 = nn.Sequential(
#         #     WTConv2d(dim,dim,kernel_size=kernel_size,stride=1,wt_levels=3),
#         #     nn.Conv2d(dim,dim,kernel_size=1)
#         # )
#         self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
#         self.layer_scale = layer_scale
#         if layer_scale is not None and type(layer_scale) in [int, float]:
#             self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
#             self.layer_scale = True
#         else:
#             self.layer_scale = False
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         b,c,t,h,w= x.size()
#         x = x.permute(0,2,1,3,4).contiguous().reshape(b*t,c,h,w)
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.act1(x)
#         x = self.conv2(x)
#         x = self.norm2(x)
#         x = input + self.drop_path(x).view(b,t,c,h,w).permute(0,2,1,3,4).contiguous()
#         return x

# class MambaSCI(nn.Module):

#     def __init__(
#         self,
#         pretrained,
#         spatial_dims: int = 3,
#         init_filters: int = 16,
#         in_channels: int = 1,
#         out_channels: int = 1,
#         n_frames: int=8,
#         dropout_prob: float | None = None,
#         act: tuple | str = ("RELU", {"inplace": True}),
#         norm: tuple | str = ("GROUP", {"num_groups": 8}),
#         norm_name: str = "",
#         num_groups: int = 8,
#         use_conv_final: bool = True,
#         blocks_down: tuple = (2,4,4,6),
#         blocks_up: tuple = (1, 1, 1),
#         upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        
#     ):
#         super().__init__()

#         if spatial_dims not in (2, 3):
#             raise ValueError("`spatial_dims` can only be 2 or 3.")

#         self.spatial_dims = spatial_dims
#         self.init_filters = init_filters
#         self.in_channels = in_channels
#         self.blocks_down = blocks_down
#         self.blocks_up = blocks_up
#         self.dropout_prob = dropout_prob
#         self.act = act  # input options
#         self.act_mod = get_act_layer(act)
#         self.out_channels = out_channels
#         self.n_frames = n_frames
#         if norm_name:
#             if norm_name.lower() != "group":
#                 raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
#             norm = ("group", {"num_groups": num_groups})
#         self.norm = norm
#         self.upsample_mode = UpsampleMode(upsample_mode)
#         self.use_conv_final = use_conv_final
#         self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)
#         self.down_layers = self._make_down_layers()
#         self.up_layers, self.up_samples = self._make_up_layers()
#         self.conv_final = self._make_final_conv(out_channels)
        

#         if isinstance(pretrained, str):
#             logger = get_root_logger()
#             load_checkpoint(self, pretrained, strict=None, logger=logger)
#         # self.net = BasicVSRPlusPlus()

# #         self.conv2 = nn.Sequential(

# #         nn.Conv3d(init_filters, init_filters*2, kernel_size=3, stride=1, padding=1),
# #         nn.LeakyReLU(inplace=True),
# #         nn.Conv3d(init_filters*2, init_filters, kernel_size=3, stride=1,padding=1),
# #         nn.LeakyReLU(inplace=True),
# #         nn.Conv3d(init_filters, 3, kernel_size=1, stride=1,padding=0),
# # )

# #         self.conv2 = nn.Sequential(
# #         nn.Conv3d(init_filters, init_filters//2, kernel_size=3, stride=1, padding=1),
# #         nn.LeakyReLU(inplace=True),
# #         nn.Conv3d(init_filters//2, init_filters//4, kernel_size=3, stride=1, padding=1),
# #         nn.LeakyReLU(inplace=True),
# #         nn.Conv3d(init_filters//4, 3, kernel_size=1, stride=1),
# # )


#         if dropout_prob is not None:
#             self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

#     def _make_down_layers(self):
#         down_layers = nn.ModuleList()
#         blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
#         for i, item in enumerate(blocks_down):
#             layer_in_channels = filters * 2**i
#             if (i==0 or i==1):
#                 downsample_mamba = (
#                 get_mamba_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, self.n_frames, stride=2)
#                 if i > 0
#                 else nn.Identity()
#             )
#                 down_layer = nn.Sequential(
#                     downsample_mamba,*[ConvBlock(dim=layer_in_channels) for _ in range(item)]
#                 )
#                 down_layers.append(down_layer)
#             else:
#                 downsample_mamba = (
#                 get_mamba_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, self.n_frames, stride=2)
#                 if i > 0
#                 else nn.Identity()
#             )
#                 down_layer = nn.Sequential(
#                 downsample_mamba, *[ResMambaBlock(spatial_dims, layer_in_channels, self.n_frames ,norm=norm, act=self.act) for _ in range(item)]
#             )
#                 down_layers.append(down_layer)
#         return down_layers

#     def _make_up_layers(self):
#         up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
#         upsample_mode, blocks_up, spatial_dims, filters, norm = (
#             self.upsample_mode,
#             self.blocks_up,
#             self.spatial_dims,
#             self.init_filters,
#             self.norm,
#         )
#         n_up = len(blocks_up)
#         for i in range(n_up):
#             sample_in_channels = filters * 2 ** (n_up - i)
#             up_layers.append(
#                 nn.Sequential(
#                     *[
#                         ResUpBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
#                         for _ in range(blocks_up[i])
#                     ]
#                 )
#             )
#             up_samples.append(
#                 nn.Sequential(
#                     *[
#                         get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
#                         get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
#                     ]
#                 )
#             )
#         return up_layers, up_samples

#     def _make_final_conv(self, out_channels: int):
#         return nn.Sequential(
#             get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
#             self.act_mod,
#             get_dwconv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
#         )

#     def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
#         # print(x.shape)
#         x = self.convInit(x)

#         down_x = []
#         down_x.append(x)

#         if self.dropout_prob is not None:
#             x = self.dropout(x)
#         # down_x = []

#         for down in self.down_layers:
#             x = down(x)
#             down_x.append(x)

#         return x, down_x

#     def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
#         # c_list = [self.init_filters*4 , self.init_filters*2,self.init_filters]
#         # self.scab = SC_Att_Bridge(c_list)
#         # # self.scab = SC_Att_Bridge()
#         # t1, t2, t3 = self.scab(down_x[1],down_x[2],down_x[3])
#         # for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
#         #     if i == 0:
#         #         x = up(x) + t1
#         #         x = upl(x)
#         #     elif i == 1:
#         #         x = up(x) + t2
#         #         x = upl(x)
#         #     elif i == 2:
#         #         x = up(x) + t3
#                 # x = upl(x)
#         for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
#             x = up(x) + down_x[i + 1]
#             x = upl(x)
#         if self.use_conv_final:
    
#             x  = x + down_x[-1]
#             #x = self.conv2(x)
#             x = self.conv_final(x)
            
#             # x = self.net(x)
#         return x

#     def forward_features(self, x: torch.Tensor) -> torch.Tensor:

#         x, down_x = self.encode(x)
#         down_x.reverse()
#         x = self.decode(x, down_x)
#         return x
    
#     # def forward(self,y,Phi,Phi_s):
#     #     out_list = []
#     #     x = At(y,Phi)
#     #     yb = A(x,Phi)
#     #     x = x + At(torch.div(y-yb,Phi_s),Phi)
#     #     x = x.unsqueeze(1)

#     #     x = self.forward_features(x)
        
#     #     out_list.append(x)
#     #     return out_list 
#     def forward(self,x):
#         x = self.forward_features(x)
#         return x 

# if __name__ == '__main__':
#     x = torch.randn(1,1,8,256,256)
#     net = MambaSCI()
#     ouy = net(x)
#     print(out.)