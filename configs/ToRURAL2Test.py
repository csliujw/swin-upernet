from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er

"""用Test generate pseudo"""

TARGET_SET = 'RURAL'

source_dir = dict(
    image_dir=[
        './LoveDA/Train/Urban/images_png/',
    ],
    mask_dir=[
        './LoveDA/Train/Urban/masks_png/',
    ],
)

target_dir = dict(
    image_dir=[
        './LoveDA/Test/Rural/images_png/',
    ],
)

eval_dir = dict(
    image_dir=[
        './LoveDA/Val/Rural/images_png/',
    ],
    mask_dir=[
        './LoveDA/Val/Rural/masks_png/',
    ],
)
test_dir = dict(
    image_dir=[
        './LoveDA/Test/Rural/images_png/',
    ]
)
# source data for train
SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
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
# target data for generate pseudo and train
TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['image_dir'],
    transforms=Compose([
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
# data for eval
EVAL_DATA_CONFIG = dict(
    image_dir=eval_dir['image_dir'],
    mask_dir=eval_dir['mask_dir'],
    transforms=Compose([
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
    training=False,
    batch_size=4,
    num_workers=4,
)

TEST_DATA_CONFIG = dict(
    image_dir=test_dir['image_dir'],
    mask_dir=test_dir['image_dir'],
    transforms=Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=4,
    num_workers=4,
)
