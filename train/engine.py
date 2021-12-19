import random
from ever.core.iterator import Iterator
import ever as er
from torch.autograd import Variable
import mmseg
import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.nn.utils import clip_grad
from torch.nn.modules import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
from models.SwinUperNet import SwinUperNet
from models.discriminator import get_fc_discriminator,FCDiscriminator
# from utils.func import prob_2_entropy, bce_loss
from utils.tools import adjust_learning_rate, COLOR_MAP, print_losses, loss_calc, adjust_learning_rate_D,bce_loss
import cv2 as cv
from utils.FocalLoss import FocalLoss

# Setup random seed
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
interp = torch.nn.Upsample(size=(512, 512), mode='bilinear',
                           align_corners=True)

print("Device: %s" % device)
writer = SummaryWriter('./log')

"""
def ad_train_1(model, max_iter, cfg, train_loader, target_loader,logger):
    criterion = CrossEntropyLoss(ignore_index=cfg.IGNORE_LABEL,reduction='mean')
    model.to(device)
    model.train()
    cudnn.benchmark = True
    cudnn.enabled = True

    # # =====================================
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load('/home/qq/code/liujiawei/MySwin/40000.pth')
    # # 将pretrained_dict里不属于model_dict的键剔除掉
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 更新现有的model_dict
    # model_dict.update(pretrained_dict)
    # # 加载我们真正需要的state_dict
    # model.load_state_dict(model_dict)
    # # =====================================


    # ===================== AD =====================
    # DISCRIMINATOR NETWORK
    # only feature-level
    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=7)
    d_main.train()
    d_main.to(device)
    d_aux = get_fc_discriminator(num_classes=7)
    d_aux.train()
    d_aux.to(device)
    # 512 *  512
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.LEARNING_RATE,betas=(0.9,0.99),weight_decay=cfg.WEIGHT_DECAY)
    optimizer_d_main = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, betas=(0.9, 0.99),
                                  weight_decay=cfg.WEIGHT_DECAY)
    optimizer_d_aux = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, betas=(0.9, 0.99),
                                  weight_decay=cfg.WEIGHT_DECAY)
    
    # labels for adversarial training
    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(train_loader)
    targetloader_iter = enumerate(target_loader)
    for cur_iters in range(1,max_iter+1):
        # =====  Train  =====
        optimizer.zero_grad()
        optimizer_d_main.zero_grad()
        optimizer_d_aux.zero_grad()

        lr = adjust_learning_rate(optimizer, cur_iters, cfg)
        adjust_learning_rate(optimizer_d_main, cur_iters, cfg)
        adjust_learning_rate(optimizer_d_aux, cur_iters, cfg)

        # UDA Training
        # only train seg net. Don't accumulate grads in disciminators
        for param in d_main.parameters():
            param.requires_grad = False
        for param in d_aux.parameters():
            param.requires_grad = False
        # take next batch
        next = trainloader_iter.__next__()[1]
        images, labels = next[0].to(device, dtype=torch.float32), next[1]['cls'].to(device, dtype=torch.long)
        pred_src_main, pred_src_aux = model(images)
        pred_src_main = interp(pred_src_main)
        pred_src_aux = interp(pred_src_aux)

        loss_seg_src_main = criterion(pred_src_main, labels)
        loss_seg_src_aux = criterion(pred_src_aux, labels)
        loss = (loss_seg_src_main + 0.1 * loss_seg_src_aux)  # 訓練分割器
        loss.backward()


        # adversarial training ot fool the discriminator
        target_image = targetloader_iter.__next__()[1][0]
        pred_trg_main, pred_trg_aux = model(target_image.to(device, dtype=torch.float32))
        pred_trg_main = interp(pred_trg_main)
        pred_trg_aux = interp(pred_trg_aux)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux, dim=1)))
        # 計算 target 和 source_label 的 loss，越小越好
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss_adv_trg_aux = bce_loss(d_out_aux, source_label)

        loss = (cfg.TRAIN_LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN_LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss = loss
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source  current
        pred_src_aux = pred_src_aux.detach()
        d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux, dim=1)))
        loss_d_aux = bce_loss(d_out_aux, source_label)
        loss_d_aux = loss_d_aux / 2
        loss_d_aux.backward()

        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main, dim=1)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        pred_trg_aux = pred_trg_aux.detach()
        d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux, dim=1)))
        loss_d_aux = bce_loss(d_out_aux, target_label)
        loss_d_aux = loss_d_aux / 2
        loss_d_aux.backward()

        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main, dim=1)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        optimizer_d_aux.step()
        optimizer_d_main.step()
        if cur_iters % 5 == 0:
            current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                              'loss_seg_src_main': loss_seg_src_main,
                              'loss_adv_trg_main': loss_adv_trg_main,  # 越小越好
                              'loss_d_main': loss_d_main,
                              'loss_adv_trg_aux': loss_adv_trg_aux,  # 越小越好
                              'loss_d_aux': loss_d_aux}  # 越大越好
            writer.add_scalar('info/lr', lr, cur_iters)
            writer.add_scalar('info/total_loss', loss, cur_iters)
            print_losses(current_losses, cur_iters, logger)
        if cur_iters % cfg.EVAL_EVERY == 0:
            torch.save(model.state_dict(), f'./model_{cur_iters}.pth')
            torch.save(d_main.state_dict(), f'./model_{cur_iters}_D_main.pth')
            torch.save(d_main.state_dict(), f'./model_{cur_iters}_D_aux.pth')
            valation(model, device, target_loader, cfg, logger)
"""



def ad_train_seg_singal(model, max_iter, cfg, train_loader, target_loader, logger):
    model.train()
    model.cuda()
    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)
    # init D
    model_D1 = FCDiscriminator(cfg.NUM_CLASSES)

    model_D1.train()
    model_D1.cuda()

    trainloader_iter = Iterator(train_loader)
    targetloader_iter = Iterator(target_loader)

    epochs = cfg.NUM_STEPS_STOP / len(train_loader)
    logger.info('epochs ~= %.3f' % epochs)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()

    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    source_label = 0
    target_label = 1
    # loss_calc = FocalLoss()
    # valation(model, device, target_loader, cfg, logger)
    for i_iter in tqdm(range(max_iter)):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        optimizer.zero_grad()
        optimizer_D1.zero_grad()
        G_lr = adjust_learning_rate(optimizer, i_iter, cfg)
        D_lr = adjust_learning_rate_D(optimizer_D1, i_iter, cfg)

        for sub_i in range(cfg.ITER_SIZE):
            # train G
            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False
            # train with source

            batch = trainloader_iter.next()
            images, labels = batch[0]
            images = Variable(images).cuda()

            pred1, _ = model(images)
            pred1 = interp(pred1)

            loss_seg1 = loss_calc(pred1, labels['cls'].cuda())
            loss = loss_seg1

            # proper normalization
            loss = loss / cfg.ITER_SIZE
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / cfg.ITER_SIZE

            # train with target
            batch = targetloader_iter.next()
            images, labels = batch[0]
            images = Variable(images).cuda()

            pred_target1, _ = model(images)
            pred_target1 = interp(pred_target1)

            D_out1 = model_D1(F.softmax(pred_target1,dim=1))

            loss_adv_target1 = utils.tools.bce_loss(D_out1,
                                       Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

            loss = cfg.LAMBDA_ADV_TARGET1  * loss_adv_target1
            loss = loss / cfg.ITER_SIZE
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / cfg.ITER_SIZE

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()

            D_out1 = model_D1(F.softmax(pred1,dim=1))

            loss_D1 = utils.tools.bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

            loss_D1 = loss_D1 / cfg.ITER_SIZE / 2

            loss_D1.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()

            # train with target
            pred_target1 = pred_target1.detach()

            D_out1 = model_D1(F.softmax(pred_target1,dim=1))

            loss_D1 = utils.tools.bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())

            loss_D1 = loss_D1 / cfg.ITER_SIZE / 2

            loss_D1.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()

        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D1.parameters()), max_norm=35, norm_type=2)

        optimizer.step()
        optimizer_D1.step()

        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            logger.info(
                'iter = %d loss_seg1 = %.3f  loss_adv1 = %.3f,  loss_D1 = %.3f  G_lr = %.5f D_lr = %.5f' % (
                    i_iter, loss_seg_value1, loss_adv_target_value1, loss_D_value1, G_lr, D_lr)
            )

        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            torch.save(model.state_dict(), f'./model_{i_iter}.pth')
            torch.save(model_D1.state_dict(), f'./model_{i_iter}_D_1.pth')
            valation(model, device, target_loader, cfg, logger)
            break

        if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), f'./model_{i_iter}.pth')
            torch.save(model_D1.state_dict(), f'./model_{i_iter}_D_1.pth')
            valation(model, device, target_loader, cfg, logger)
            model.train()


def ad_train_seg_mutil(model, max_iter, cfg, train_loader, target_loader, logger):
    model.train()
    model.cuda()
    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)
    # init D
    model_D1 = FCDiscriminator(cfg.NUM_CLASSES)
    model_D2 = FCDiscriminator(cfg.NUM_CLASSES)

    model_D1.train()
    model_D1.cuda()

    model_D2.train()
    model_D2.cuda()

    model.load_state_dict(torch.load('model_12000.pth'))
    model_D1.load_state_dict(torch.load('model_12000_D_1.pth'))
    model_D2.load_state_dict(torch.load('model_12000_D_2.pth'))
    trainloader_iter = Iterator(train_loader)
    targetloader_iter = Iterator(target_loader)

    epochs = cfg.NUM_STEPS_STOP / len(train_loader)
    logger.info('epochs ~= %.3f' % epochs)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()

    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = torch.optim.Adam(model_D2.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    source_label = 0
    target_label = 1
    loss_calc = FocalLoss()
    # valation(model, device, target_loader, cfg, logger)
    for i_iter in tqdm(range(max_iter)):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        G_lr = adjust_learning_rate(optimizer, i_iter, cfg)
        D_lr = adjust_learning_rate_D(optimizer_D1, i_iter, cfg)
        adjust_learning_rate_D(optimizer_D2, i_iter, cfg)

        for sub_i in range(cfg.ITER_SIZE):
            # train G
            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False
            # train with source

            batch = trainloader_iter.next()
            images, labels = batch[0]
            images = Variable(images).cuda()

            pred1, pred2 = model(images)
            pred1 = interp(pred1)
            pred2 = interp(pred2)

            loss_seg1 = loss_calc(pred1, labels['cls'].cuda())
            loss_seg2 = loss_calc(pred2, labels['cls'].cuda())
            loss = loss_seg1 + cfg.LAMBDA_SEG * loss_seg2

            # proper normalization
            loss = loss / cfg.ITER_SIZE
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / cfg.ITER_SIZE
            loss_seg_value2 += loss_seg2.data.cpu().numpy() / cfg.ITER_SIZE

            # train with target
            batch = targetloader_iter.next()
            images, labels = batch[0]
            images = Variable(images).cuda()

            pred_target1, pred_target2 = model(images)
            pred_target1 = interp(pred_target1)
            pred_target2 = interp(pred_target2)

            D_out1 = model_D1(F.softmax(pred_target1, dim=1))
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))

            loss_adv_target1 = utils.tools.bce_loss(D_out1,
                                                    Variable(torch.FloatTensor(D_out1.data.size()).fill_(
                                                        source_label)).cuda())

            loss_adv_target2 = utils.tools.bce_loss(D_out2,
                                                    Variable(torch.FloatTensor(D_out2.data.size()).fill_(
                                                        source_label)).cuda())

            loss = cfg.LAMBDA_ADV_TARGET1 * loss_adv_target1 + cfg.LAMBDA_ADV_TARGET2 * loss_adv_target2
            loss = loss / cfg.ITER_SIZE
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / cfg.ITER_SIZE
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / cfg.ITER_SIZE

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()

            D_out1 = model_D1(F.softmax(pred1, dim=1))
            D_out2 = model_D2(F.softmax(pred2, dim=1))

            loss_D1 = utils.tools.bce_loss(D_out1,
                                           Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())

            loss_D2 = utils.tools.bce_loss(D_out2,
                                           Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())

            loss_D1 = loss_D1 / cfg.ITER_SIZE / 2
            loss_D2 = loss_D2 / cfg.ITER_SIZE / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            D_out1 = model_D1(F.softmax(pred_target1, dim=1))
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))

            loss_D1 = utils.tools.bce_loss(D_out1,
                                           Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())

            loss_D2 = utils.tools.bce_loss(D_out2,
                                           Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())

            loss_D1 = loss_D1 / cfg.ITER_SIZE / 2
            loss_D2 = loss_D2 / cfg.ITER_SIZE / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D1.parameters()), max_norm=35, norm_type=2)
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D2.parameters()), max_norm=35, norm_type=2)

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        if i_iter % 50 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            logger.info(
                'iter = %d loss_seg1 = %.3f loss_seg2 = %.3f loss_adv1 = %.3f, loss_adv2 = %.3f loss_D1 = %.3f loss_D2 = %.3f G_lr = %.5f D_lr = %.5f' % (
                    i_iter, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2,
                    loss_D_value1, loss_D_value2, G_lr, D_lr)
            )

        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            torch.save(model.state_dict(), f'./model_{i_iter}.pth')
            torch.save(model_D1.state_dict(), f'./model_{i_iter}_D_1.pth')
            torch.save(model_D2.state_dict(), f'./model_{i_iter}_D_2.pth')
            valation(model, device, target_loader, cfg, logger)
            break

        if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), f'./model_{i_iter}.pth')
            torch.save(model_D1.state_dict(), f'./model_{i_iter}_D_1.pth')
            torch.save(model_D2.state_dict(), f'./model_{i_iter}_D_2.pth')
            valation(model, device, target_loader, cfg, logger)
            model.train()


def train_single(model, max_iter, cfg, train_loader, val_loader, logger):
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.LEARNING_RATE,betas=(0.9,0.99),weight_decay=cfg.WEIGHT_DECAY)
    criterion = CrossEntropyLoss(ignore_index=cfg.IGNORE_LABEL,reduction='mean')

    model.to(device)
    model.train()
    cur_iters = 0

    length = len(train_loader)

    while cur_iters < max_iter:
        for batch in train_loader:
            cur_iters += 1
            images, labels = batch[0].to(device, dtype=torch.float32),batch[1]['cls'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            primary,_ = model(images)
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
                    print("===================start target valation===================")
                    valation(model, device, val_loader,cfg,logger)
                    if cur_iters % cfg.TRAIN_EVERY == 0:
                        print("===================start train valation===================")
                        valation(model, device, train_loader,cfg,logger)

def train_mutil(model, max_iter, cfg, train_loader, val_loader,logger):
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.LEARNING_RATE,betas=(0.9,0.99),weight_decay=cfg.WEIGHT_DECAY)
    criterion = CrossEntropyLoss(ignore_index=cfg.IGNORE_LABEL,reduction='mean')

    model.to(device)
    model.train()
    cur_iters = 0

    length = len(train_loader)

    while cur_iters < max_iter:
        for batch in train_loader:
            cur_iters += 1
            images, labels = batch[0].to(device, dtype=torch.float32),batch[1]['cls'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            primary,aux = model(images)
            primary = interp(primary)
            aux = interp(aux)
            loss_primary = criterion(primary, labels)
            loss_aux = criterion(aux, labels)
            lr = adjust_learning_rate(optimizer, cur_iters, cfg)
            loss = loss_primary + 0.4*loss_aux
            loss.backward()
            optimizer.step()
            writer.add_scalar('info/lr', lr, cur_iters)
            writer.add_scalar('info/total_loss', loss, cur_iters)
            if cur_iters % 10 == 0:
                logger.info("Epoch %d, Itrs %d/%d, decode.loss=%f aux.loss=%f,;lr=%f" % (cur_iters/length+1 , cur_iters, max_iter,loss_primary.data,loss_aux.data,lr))
                if cur_iters % cfg.EVAL_EVERY == 0:
                    torch.save(model.state_dict(), f"./{cur_iters}.pth")
                    print("===================start target valation===================")
                    valation(model, device, val_loader,cfg,logger)
                    if cur_iters % cfg.TRAIN_EVERY == 0:
                        print("===================start train valation===================")
                        valation(model, device, train_loader,cfg,logger)

@torch.no_grad()
def valation(model, device, val_loader,cfg,logger):
    model.eval()
    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
    with torch.no_grad():
        for ret, ret_gt in tqdm(val_loader):
            ret = ret.to(device)
            cls,_ = model(ret)
            cls = interp(cls)
            cls = cls.argmax(dim=1).cpu().numpy()
            cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
            mask = cls_gt >= 0
            y_true = cls_gt[mask].ravel()
            y_pred = cls[mask].ravel()
            metric_op.forward(y_true, y_pred)
    metric_op.summary_all()
    model.train()

"""
def predict(model, test_loader,logger):
    model.load_state_dict(torch.load('/home/qq/code/liujiawei/MySwin/40000.pth'))

    model.to(device)
    model.train()
    upper = torch.nn.Upsample(size=(1024, 1024), mode='bilinear',
                               align_corners=True)
    with torch.no_grad():
        for batch in test_loader:
            images, names = batch[0].to(device, dtype=torch.float32),batch[1]['fname']
            primary,aux = model(images)
            primary = upper(primary)
            primary = primary.argmax(dim=1).cpu().numpy()
            for index in range(len(primary)):
                cv.imwrite(f"/home/qq/code/liujiawei/MySwin/train/pred/{names[index]}", primary[index])
            # aux = interp(aux)
"""

if __name__ == '__main__':
    model = SwinUperNet()