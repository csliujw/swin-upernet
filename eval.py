import logging

from data.loveda import LoveDALoader

logger = logging.getLogger(__name__)
from utils.tools import *
from ever.util.param_util import count_model_parameters

eval_interp = torch.nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)


def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None):
    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict,  strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)
    model.eval()
    print(cfg.EVAL_DATA_CONFIG)
    eval_dataloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    metric_op = er.metric.PixelMetric(len(COLOR_MAP.keys()), logdir=cfg.SNAPSHOT_DIR, logger=logger)
    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)[0]
            cls = eval_interp(cls)
            cls = cls.argmax(dim=1).cpu().numpy()

            cls_gt = ret_gt['cls'].cpu().numpy().astype(np.int32)
            mask = cls_gt >= 0

            y_true = cls_gt[mask].ravel()
            y_pred = cls[mask].ravel()
            metric_op.forward(y_true, y_pred)
    metric_op.summary_all()
    torch.cuda.empty_cache()
