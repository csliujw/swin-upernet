import random

import ever as er
import numpy as np
import torch
from torch.nn.modules import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.SwinUperNet import SwinUperNet
from utils.tools import adjust_learning_rate, COLOR_MAP

# Setup random seed
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
interp = torch.nn.Upsample(size=(512, 512), mode='bilinear',
                           align_corners=True)

print("Device: %s" % device)
writer = SummaryWriter('./log')


def train(model, max_iter, cfg, train_loader, val_loader,logger):
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.LEARNING_RATE,betas=(0.9,0.99),weight_decay=cfg.WEIGHT_DECAY)
    criterion = CrossEntropyLoss(ignore_index=cfg.IGNORE_LABEL,reduction='mean')

    model.to(device)
    model.train()
    cur_iters = 0

    length = len(train_loader)
    # valation(model, device, val_loader, cfg, logger)
    # valation(model, device, train_loader, cfg, logger)
    while cur_iters < max_iter:
        for batch in train_loader:
            cur_iters += 1
            images, labels = batch[0].to(device, dtype=torch.float32),batch[1]['cls'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            primary = model(images)
            primary = interp(primary)

            loss_primary = criterion(primary, labels)
            lr = adjust_learning_rate(optimizer, cur_iters, cfg)
            loss = loss_primary
            loss.backward()
            optimizer.step()
            writer.add_scalar('info/lr', lr, cur_iters)
            writer.add_scalar('info/total_loss', loss, cur_iters)
            if cur_iters % 10 == 0:
                logger.info("Epoch %d, Itrs %d/%d, decode.loss=%f ;lr=%f" % (cur_iters/length+1 , cur_iters, max_iter,loss_primary.data,lr))
                if cur_iters % cfg.EVAL_EVERY == 0:
                    torch.save(model.state_dict(), f"./{cur_iters}.pth")
                    valation(model, device, val_loader,cfg,logger)
                    valation(model, device, train_loader,cfg,logger)


@torch.no_grad()
def valation(model, device, val_loader,cfg,logger):
    model.eval()
    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
    with torch.no_grad():
        for ret, ret_gt in tqdm(val_loader):
            ret = ret.to(torch.device(device))
            cls = model(ret)
            cls = interp(cls)
            cls = cls.argmax(dim=1).cpu().numpy()
            cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
            mask = cls_gt >= 0
            y_true = cls_gt[mask].ravel()
            y_pred = cls[mask].ravel()
            metric_op.forward(y_true, y_pred)
    metric_op.summary_all()
    model.train()

if __name__ == '__main__':
    model = SwinUperNet()
    lk = []
    for k,v in model.state_dict().items():
        lk.append(k)
    s = torch.load('/home/qq/code/liujiawei/MySwin/swin_base_patch4_window7_224.pth')
    tmp = list(s['model'].keys())
    ss = [k for k in tmp]

    for i in lk:
        print(i)
    print("================")
    print("================")
    print("================")
    for i in ss:
        print(i)