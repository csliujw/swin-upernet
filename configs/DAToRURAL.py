from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale, Resize
from albumentations import OneOf, Compose
import ever as er

BASE_DIR = "/media/qq/0CBE052B0CBE052B/dataset/LoveDA/jieya"

TARGET_SET = 'RURAL'

source_dir = dict(
    image_dir=[
        BASE_DIR+'/img_dir/train/',
    ],
    mask_dir=[
        BASE_DIR+'/ann_dir/train/',
    ],
)
target_dir = dict(
    image_dir=[
        BASE_DIR+'/img_dir/val/',
    ],
    mask_dir=[
        BASE_DIR+'/ann_dir/val/',
    ],
)
test_dir = dict(
    image_dir=[
        BASE_DIR+'/img_dir/test/',
    ],
)

SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
        Resize(2048, 512,p=0.5),
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.5),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=4,
    num_workers=4,
)


TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        Resize(2048, 512,p=0.5),
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.5),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=4,
    num_workers=4,
)

EVAL_DATA_CONFIG = dict(
    image_dir=test_dir['image_dir'],
    mask_dir=target_dir['image_dir'],
    transforms=Compose([
        Resize(2048, 512, p=0.5),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=2,
    num_workers=2,
)
