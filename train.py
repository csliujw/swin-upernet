import argparse
from collections import OrderedDict

import cv2
import torch
import torch.nn.functional as F
import torch.utils.data as data
from ever.core.iterator import Iterator
from mmcv import Config

from dataset.nj import NJLoader
from models.SwinUperNet import SwinUperNet
from train.engine import train
from utils.tools import import_config, get_console_file_logger

parser = argparse.ArgumentParser(description='Run CBST methods.')

parser.add_argument('--config_path', type=str, default="st.cbst.2rural",
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)

# trainloader = NJLoader(cfg.SOURCE_DATA_CONFIG)
# trainloader_iter = Iterator(trainloader)
#
# evalloader = NJLoader(cfg.EVAL_DATA_CONFIG)
# evalloader_iter = Iterator(evalloader)
#
# targetloader = NJLoader(cfg.TARGET_DATA_CONFIG)
# targetloader_iter = Iterator(targetloader)
# batch = trainloader_iter.next()
# images_s, labels_s = batch[0]


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a segmentor')
#     parser.add_argument('--config', type=str,
#                         default='/home/qq/code/liujiawei/MySwin/train/work_dir/upernet_swin_base_window7.py',
#                         help='train config file path')
#     parser.add_argument('--seed', type=int, default=None, help='random seed')
#     args = parser.parse_args()
#
#     return args

logger = get_console_file_logger(name='MySwin', logdir=cfg.SNAPSHOT_DIR)


def main():
    # args = parse_args()
    # cfg_formfile = Config.fromfile(args.config)
    model = SwinUperNet()
    torch.cuda.empty_cache()
    trainloader = NJLoader(cfg.SOURCE_DATA_CONFIG)
    evalloader = NJLoader(cfg.TARGET_DATA_CONFIG)
    train(model, cfg.NUM_STEPS, cfg, trainloader, evalloader,logger)


# @torch.no_grad()
# def main_test():
#     interp = torch.nn.Upsample(size=(1024, 1024), mode='bilinear',
#                                align_corners=True)
#     model = SwinUperNet()
#     # print(model)
#     load_dict = torch.load(
#         '/home/qq/code/liujiawei/Swin-Transformer-UperNet/work_dirs/upernet_swin_base_patch4_window7_512x512_80k_loveda/iter_40000.pth')
#     load_dict = load_dict['state_dict']  # some weight = ['backbone.patch_embed.projection.weight']
#     weight = OrderedDict()
#     for key, value in load_dict.items():
#         weight[key] = value
#     model.load_state_dict(weight)
#     model.to('cuda')
#     model.eval()
#     dataset = LoveDASegmentation(mode='test')
#     train_loader = data.DataLoader(
#         dataset, batch_size=2, shuffle=False, num_workers=16, drop_last=True)
#     with torch.no_grad():
#         for i, batch in enumerate(train_loader):
#             images = batch.to('cuda')
#             main_seg = model(images)
#             main_seg = interp(main_seg)
#             pred = F.softmax(main_seg, dim=1)
#             pred = pred.argmax(dim=1).cpu().numpy()
#             for index in range(pred.shape[0]):
#                 cv2.imwrite(f"dataset/img/{dataset.images[i].split('/')[-1]}", pred[index])


if __name__ == '__main__':
    main()
