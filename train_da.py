import argparse

import torch
import torch

from data.loveda import LoveDALoader
from module.SwinUperNet import SwinUperNet
from train.engine import train_single, predict
from utils.tools import import_config, get_console_file_logger
from train.engine import ad_train_advent_single
from utils.tools import import_config, get_console_file_logger

parser = argparse.ArgumentParser(description='Run CBST methods.')

parser.add_argument('--config_path', type=str, default="st.cbst.da2rural",
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)

logger = get_console_file_logger(name='DA-Swin', logdir=cfg.SNAPSHOT_DIR)


def main():
    model = SwinUperNet()
    torch.cuda.empty_cache()
    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    targetloader = LoveDALoader(cfg.TARGET_DATA_CONFIG)
    ad_train_advent_single(model, cfg.NUM_STEPS, cfg, trainloader, targetloader,logger)

if __name__ == '__main__':
    main()
