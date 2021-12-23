import torch.nn as nn

from module.FCNHead import FCNHead
from module.Swin import SwinTransformer
from module.UperHead import UPerHead


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
        self.auxiliary_head = FCNHead(
            in_channels=512,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
        )

    def forward(self,input):
        x = self.backbone(input)
        main_ = self.decode_head(x)
        aux_ = self.auxiliary_head(x)
        return main_,aux_ # 主分类器，辅助分类器

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