import random

import numpy as np
import torch
import torch.nn.functional as F
from ever.core.iterator import Iterator
from torch.autograd import Variable
from torch.nn.utils import clip_grad
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from models.discriminator import get_fc_discriminator, FCDiscriminator
from utils.FocalLoss import FocalLoss
from utils.tools import adjust_learning_rate, loss_calc, adjust_learning_rate_D

# Setup random seed
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
interp = torch.nn.Upsample(size=(512, 512), mode='bilinear',
                           align_corners=True)

print("Device: %s" % device)
writer = SummaryWriter('./log')

def ad_train_seg_singal(model, max_iter, cfg, train_loader, target_loader, logger):
    model.train()
    model.cuda()
    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)
    # init D
    model_D1 = get_fc_discriminator(cfg.NUM_CLASSES)

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
            images = images.cuda()

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
            images = images.cuda()

            pred_target1, _ = model(images)
            pred_target1 = interp(pred_target1)

            D_out1 = model_D1(F.softmax(pred_target1,dim=1))

            loss_adv_target1 = utils.tools.bce_loss(D_out1,torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())

            loss = cfg.LAMBDA_ADV_TARGET1 * loss_adv_target1
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

    # model.load_state_dict(torch.load('model_12000.pth'))
    # model_D1.load_state_dict(torch.load('model_12000_D_1.pth'))
    # model_D2.load_state_dict(torch.load('model_12000_D_2.pth'))
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

