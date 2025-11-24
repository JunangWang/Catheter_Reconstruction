import tyro
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

@dataclass
class Options:
    ### model
    # Unet image input size
    input_size: int = 256
    # Unet definition
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # Volume size
    output_size: int = 256

    ### dataset
    # total number of volumes
    num_volumes: int = 100
    # total number of image groups for each volume
    num_image_groups: int = 20

    ### training
    # workspace
    workspace: str = '/home/junang/qubot/checkpoints/'
    # resume
    resume: Optional[str] = None
    # train dataset path
    train_path: Optional[str] = '/home/ubuntu/giant/guixiang/dataset_tiff/'
    # field of view
    fovy: float = 20.03
    # source Pose matrix
    pose_path: str = '/home/ubuntu/giant/guixiang/dataset_tiff/images/sourcePoseMatrix.npy'
    # wandb project name
    wandb_name: Optional[str] = '100_20_image_bce_gdl_set7'
    # batch size (per-GPU)
    batch_size: int = 8
    # training epochs
    num_epochs: int = 30
    # learning rate
    lr: float = 4e-4
    # gradient clip
    gradient_clip: float = 1.0
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # mixed precision
    mixed_precision: str = 'bf16'
    # input
    ray_embedding: bool = False


# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['lrm'] = 'the default settings for LGM'
config_defaults['lrm'] = Options()

config_doc['small'] = 'small model with lower resolution Gaussians'
config_defaults['small'] = Options(
    input_size=1024,
    output_size=256,
    batch_size=1,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['big'] = 'big model with higher resolution Gaussians'
config_defaults['big'] = Options(
    input_size=512,
    down_channels= (64, 128, 256, 512, 1024, 1024), # origin
    down_attention= (False, False, False, True, True, True),
    up_channels=(1024, 1024, 512, 256, 128), # one more decoder origin
    up_attention=(True, True, True, False, False),
    output_size=512,
    wandb_name='1K_60_images_rayEmbedding_lr_pred_trained_1.2e-5', 
    resume = '/home/junang/qubot/checkpoints/best_model/model.safetensors',
    num_volumes=5000,
    num_image_groups=60,
    batch_size=4,
    num_epochs=10,
    lr=1.2e-5,
    ray_embedding=True,
    gradient_clip=1,
    gradient_accumulation_steps=1,
    mixed_precision='bf16'
)

config_doc['tiny'] = 'tiny model for ablation'
config_defaults['tiny'] = Options(
    input_size=1024, 
    down_channels=(32, 64, 128),
    down_attention=(False, False, True),
    up_channels=(128, 64),
    up_attention=(True, False),
    output_size=1024,
    batch_size=1,
    num_epochs=50,
    lr=4e-4,
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)