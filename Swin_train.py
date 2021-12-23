import argparse

import torch

from data.loveda import LoveDALoader
from module.SwinUperNet import SwinUperNet
from train.engine import train_single, predict
from utils.tools import import_config, get_console_file_logger

parser = argparse.ArgumentParser(description='Run CBST methods.')
parser.add_argument('--config_path', type=str, default="st.cbst.2rural",
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)
logger = get_console_file_logger(name='MySwin', logdir=cfg.SNAPSHOT_DIR)


def submit():
    model = SwinUperNet()
    torch.cuda.empty_cache()
    cfg.EVAL_DATA_CONFIG['image_dir'] = "/media/qq/0CBE052B0CBE052B/dataset/LoveDA/2021LoveDA/Val/Rural/images_png"
    cfg.EVAL_DATA_CONFIG['mask_dir'] = "/media/qq/0CBE052B0CBE052B/dataset/LoveDA/2021LoveDA/Val/Rural/images_png"
    testloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    predict(model,testloader)


def main():
    model = SwinUperNet()
    torch.cuda.empty_cache()
    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    targetloader = LoveDALoader(cfg.TARGET_DATA_CONFIG)
    model.load_state_dict(torch.load('0.32.pth'))
    train_single(model, cfg.NUM_STEPS, cfg, trainloader, targetloader, logger)

if __name__ == '__main__':
    submit()
