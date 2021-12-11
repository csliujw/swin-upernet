import torch.nn as nn
import torch.nn.modules
from models.Swin import SwinTransformer
from models.UperHead import UPerHead
"""
convs
   (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activate): ReLU(inplace=True)
cls_seg Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1))
"""
# class FCN(nn.Module):
#     def __init__(self,in_channels=[128, 256, 512, 1024],
#                             pool_scales=(1, 2, 3, 6),
#                             channels=512,
#                             dropout_ratio=0.1,
#                             num_classes=8):
#         super(FCN, self).__init__()
#         self.conv_128 = nn.Sequential(
#             nn.Conv2d(128,128,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=True),
#             nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True),
#             nn.ReLU())
#
#         self.conv_256 = nn.Sequential(
#             nn.Conv2d(256,128,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=True),
#             nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True),
#             nn.ReLU())
#         self.conv_512 = nn.Sequential(
#             nn.Conv2d(512,128,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=True),
#             nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True),
#             nn.ReLU())
#         self.conv_1024 = nn.Sequential(
#             nn.Conv2d(1024,128,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=True),
#             nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True),
#             nn.ReLU())
#         self.cls_seg = nn.Sequential(nn.Conv2d(channels, num_classes, kernel_size=(1, 1), stride=(1, 1)))
#         self.up = torch.nn.Upsample(size=(128, 128), mode='bilinear',align_corners=True)
#
#     def forward(self,x):
#         x1 = self.conv_128(x[0])
#         x2 = self.conv_256(x[1])
#         x3 = self.conv_512(x[2])
#         x4 = self.conv_1024(x[3])
#         x2 = self.up(x2)
#         x3 = self.up(x3)
#         x4 = self.up(x4)
#         out = torch.cat([x1, x2,x3,x4], dim=1)
#         return self.cls_seg(out)
#         """
#         2,128,128,128,
#         2,256,64,64,
#         2,512,32,32
#         2,1024,16,16
#         """

class SwinUperNet(nn.Module):
    def __init__(self):
        super(SwinUperNet,self).__init__()
        self.backbone = SwinTransformer(
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            use_abs_pos_embed=False)
        self.decode_head = UPerHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False
        )

    def forward(self,input):
        x = self.backbone(input)
        main_ = self.decode_head(x)
        return main_ # 主分类器，辅助分类器

if __name__ == '__main__':
    import torch
    device = torch.device("cuda")
    model = SwinUperNet()
    # batchsize % 2==0
    images = torch.rand(size=(2,3,512,512))
    images = images.to(device, dtype=torch.float32)
    model.to(device)
    ret1 = model(images)
    print(model)