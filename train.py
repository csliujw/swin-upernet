import argparse

import torch

from dataset.nj import NJLoader, NJPredictLoader
from models.SwinUperNet import SwinUperNet
from train.engine import train_single
from utils.tools import import_config, get_console_file_logger

parser = argparse.ArgumentParser(description='Run CBST methods.')
parser.add_argument('--config_path', type=str, default="st.cbst.2rural",
                    help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)
logger = get_console_file_logger(name='MySwin', logdir=cfg.SNAPSHOT_DIR)


def predict():
    model = SwinUperNet()
    torch.cuda.empty_cache()
    cfg.EVAL_DATA_CONFIG['image_dir'] = "/media/qq/0CBE052B0CBE052B/dataset/LoveDA/2021LoveDA/Test/Rural/images_png"
    cfg.EVAL_DATA_CONFIG['mask_dir'] = "/media/qq/0CBE052B0CBE052B/dataset/LoveDA/2021LoveDA/Test/Rural/images_png"
    testloader = NJPredictLoader(cfg.EVAL_DATA_CONFIG)
    predict(model,testloader,logger)


def main():
    model = SwinUperNet()
    torch.cuda.empty_cache()
    trainloader = NJLoader(cfg.SOURCE_DATA_CONFIG)
    valloader = NJLoader(cfg.TARGET_DATA_CONFIG)
    train_single(model, cfg.NUM_STEPS, cfg, trainloader, valloader, logger)

if __name__ == '__main__':
    main()
