import tyro
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, Literal

@dataclass
class Options:
    ### model
    # ViT image input size
    img_size: int = 512
    input_channels: int = 2
    # ViT definition
    patch_gird: Tuple[int, int] = (32, 32)
    patch_size: Tuple[int, int] = (16, 16)
    hidden_size: int = 768
    attn_num_layers: int = 12
    attn_num_heads: int = 12
    mlp_dim: int = 3072
    resnet_num_layers: Tuple[int, ...] = (3, 4, 9) 
    resnet_width_factor: int = 1

    qkv_bias: bool = True
    proj_bias: bool = True
    embed_drop: float = 0.15
    attn_drop: float = 0.1
    attn_proj_drop: float = 0.1
    mlp_drop: float = 0.1
    
    # Cup decoder
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 16)
    n_skip: int = 3
    skip_channels: Tuple[int, ...] = (512, 256, 64, 16)

    # visualize attention weights
    vis: bool = False

    # classifier
    classifier: str = "arts"
    # pred object type
    pred_obj_type: Literal['vessel', 'arts', 'wire'] = 'vessel'

    # tasks arts, seg or multi
    task: Literal['arts','seg','multi', 'mix_seg'] = 'arts'
    # fine-tune
    freeze_arts: bool = False
    ### dataset
    # total number of images
    num_img: int = 100
    real_data_scale: int = 1
    # 3d wire
    num_view: int = 2
    coord_dim: int = 3
    total_num_control_point: int = 16
    num_coords: int = 1000

    ### training
    # workspace
    workspace: str = '/home/junang/qubot/artifacts_removal/checkpoints/'
    # Resnet 50 and ViT pretrained model path
    resnet_pretrained_path: Optional[str] = None
    pretrained_path: Optional[str] =  '/home/junang/Artifacts_Removal/model/vit_checkpoint/imagenet21k_R50+ViT-B_16.npz'
    # resume
    resume: Optional[str] = None
    load_module_names: Optional[Tuple[str, ...]] = None
    # train dataset path
    train_path: Optional[str] = '/home/junang/Artifacts_Removal/data/'
    spatial_path: Optional[str] = None
    # train_path: Optional[str] = '/home/ubuntu/giant/junang/TransUnet/'
    # test dataset path
    test_path: Optional[str] = None
    

    # wandb project name
    wandb_name: Optional[str] = '1K_images_TransUnet_lr_pred_trained_1e-2'
    # batch size (per-GPU)
    batch_size: int = 24
    # training epochs
    num_epochs: int = 150
    # learning rate
    lr: float = 1e-3
    # gradient clip
    gradient_clip: float = 1.0
    # gradient accumulation
    gradient_accumulation_steps: int = 2
    # mixed precision
    mixed_precision: str = 'bf16'



# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['TransUnet'] = 'the default settings for TransUnet'
config_defaults['TransUnet'] = Options()


config_doc['mid_inference'] = 'mid model with 512x512 resolution'
config_defaults['mid_inference'] = Options(
    img_size=512,
    hidden_size=768,
    attn_num_layers=12,
    attn_num_heads=12,
    mlp_dim=3072,
    patch_gird=(32, 32),
    patch_size=(16, 16), # 32*16 = 512 = img_size
    task = 'multi',
    # pred_obj_type='arts',
    train_path= '/home/ubuntu/giant/junang/TransUnet/gear_background_2000.h5',
    test_path = '/home/ubuntu/giant/junang/TransUnet/Unsupervised_test_DIAS.h5',
    pretrained_path=None,

    resume = "/home/junang/qubot/artifacts_removal/checkpoints/real_multi_task_v2/best_model/model.safetensors",
    workspace = '/home/ubuntu/giant/junang/qubot/artifacts_removal/checkpoints/real_multi_task_v2',
    num_img=2000,
    batch_size=24,
    num_epochs=100,
    lr=1.0e-3,

    mixed_precision='bf16',
)



config_doc['multi_mid_mixed_wire'] = 'mid model with 512x512 resolution trained by wire mixed dataset'
config_defaults['multi_mid_mixed_wire'] = Options(
    img_size=512,
    hidden_size=768,
    attn_num_layers=12,
    attn_num_heads=12,
    mlp_dim=3072,
    patch_gird=(32, 32),
    patch_size=(16, 16), # 32*16 = 512 = img_size
    task = 'multi',
    pred_obj_type= 'wire',
    pretrained_path=None,
    train_path= '/home/ubuntu/giant/junang/TransUnet/shuffled_background_wire_v2_16666.h5+real_background_wire_v2_16666.h5+gear_background_wire_v2_16666.h5+shuffled_background_wire_v2_12000.h5+real_background_wire_v2_12000.h5+gear_background_wire_v2_12000.h5',
    test_path = None,

    # resume = "/home/junang/qubot/artifacts_removal/checkpoints/best_model/model.safetensors",
    workspace = '/home/junang/qubot/artifacts_removal/checkpoints/wire_mixed_multi_task_3d_72000_v1',
    wandb_name='TransUnet_lr_1e-3_wd_1e-1_Adam_18_batch_data_v2_wire_multi_mixed_3d_72000_v1', 
    num_img=72000,
    batch_size=9,
    num_epochs=50,
    lr=5.0e-3,
    total_num_control_point = 20,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['multi_mid_mixed_wire_2d'] = 'mid model with 512x512 resolution trained by wire mixed dataset'
config_defaults['multi_mid_mixed_wire_2d'] = Options(
    task = 'multi',
    pred_obj_type= 'wire',
    coord_dim = 2,
    num_view = 1,
    pretrained_path=None,
    train_path= '/home/ubuntu/giant/junang/TransUnet/shuffled_background_wire_v2_16666.h5+real_background_wire_v2_16666.h5+gear_background_wire_v2_16666.h5+shuffled_background_wire_v2_12000.h5+real_background_wire_v2_12000.h5+gear_background_wire_v2_12000.h5',
    test_path = None,

    # resume = "/home/junang/qubot/artifacts_removal/checkpoints/best_model/model.safetensors",
    workspace = '/home/junang/qubot/artifacts_removal/checkpoints/wire_mixed_multi_task_3d_72000_2d_v1',
    wandb_name='TransUnet_lr_1e-3_wd_1e-1_Adam_18_batch_data_v2_wire_multi_mixed_72000_2d_v1', 
    num_img=72000,
    batch_size=24,
    num_epochs=40,
    lr=5.0e-4,
    total_num_control_point = 20,
    gradient_accumulation_steps=1,
    mixed_precision='bf16',
)

config_doc['multi_mid_mixed_wire_fine_tune'] = 'mid model with 512x512 resolution trained by wire mixed dataset'
config_defaults['multi_mid_mixed_wire_fine_tune'] = Options(
    task = 'multi',
    pred_obj_type= 'wire',
    pretrained_path=None,
    train_path= '/home/ubuntu/giant/junang/TransUnet/shuffled_background_wire_v2_16666.h5+real_background_wire_v2_16666.h5+gear_background_wire_v2_16666.h5+shuffled_background_wire_v2_12000.h5+real_background_wire_v2_12000.h5+gear_background_wire_v2_12000.h5',
    # spatial_path = '/home/ubuntu/giant/guixiang/dataset_520/coords_50000.h5',
    test_path = None,

    resume = "/home/junang/qubot/artifacts_removal/checkpoints/wire_mixed_multi_task_3d_72000_v1/best_model/model.safetensors",
    freeze_arts = True,
    # workspace = '/home/junang/qubot/artifacts_removal/checkpoints/wire_mixed_multi_task_3d_16666_v3',
    workspace = '/home/junang/qubot/artifacts_removal/checkpoints/wire_mixed_multi_task_3d_16666_v6_fine_tune_v3',
    wandb_name='TransUnet_lr_1e-3_wd_1e-1_Adam_20_batch_data_v2_wire_multi_mixed_3d_16666_v6_fine_tune', 
    num_img=720,
    batch_size=24,
    num_epochs=15,
    lr=4.0e-4,
    total_num_control_point = 20,
    gradient_accumulation_steps=1,

    mixed_precision='bf16',
)


config_doc['multi_mid_mixed_wire_fine_tune_2d'] = 'mid model with 512x512 resolution trained by wire mixed dataset'
config_defaults['multi_mid_mixed_wire_fine_tune_2d'] = Options(
    task = 'multi',
    pred_obj_type= 'wire',
    coord_dim = 2,
    num_view = 1,
    pretrained_path=None,
    train_path= '/home/ubuntu/giant/junang/TransUnet/shuffled_background_wire_v2_16666.h5+real_background_wire_v2_16666.h5+gear_background_wire_v2_16666.h5+shuffled_background_wire_v2_12000.h5+real_background_wire_v2_12000.h5+gear_background_wire_v2_12000.h5',
    # spatial_path = '/home/ubuntu/giant/guixiang/dataset_520/coords_50000.h5',
    test_path = None,

    resume = "/home/junang/qubot/artifacts_removal/checkpoints/wire_mixed_multi_task_3d_72000_v1/best_model/model.safetensors",
    load_module_names = ('ViT.transformer', 'ViT.decoder_arts', 'ViT.arts_head', 'ViT.seg_head'),
    freeze_arts = True,
    # workspace = '/home/junang/qubot/artifacts_removal/checkpoints/wire_mixed_multi_task_3d_16666_v3',
    workspace = '/home/junang/qubot/artifacts_removal/checkpoints/wire_mixed_multi_task_3d_16666_v6_fine_tune_v2_2d',
    wandb_name='TransUnet_lr_1e-3_wd_1e-1_Adam_20_batch_data_v2_wire_multi_mixed_3d_16666_v6_fine_tune_2d', 
    num_img=72000,
    batch_size=24,
    num_epochs=15,
    lr=4.0e-4,
    total_num_control_point = 20,
    gradient_accumulation_steps=1,

    mixed_precision='bf16',
)

config_doc['inference_vessel'] = 'inference model with 512x512 resolution trained by wire mixed dataset'
config_defaults['inference_vessel'] = Options(
    img_size=512,
    hidden_size=768,
    attn_num_layers=12,
    attn_num_heads=12,
    mlp_dim=3072,
    patch_gird=(32, 32),
    patch_size=(16, 16), # 32*16 = 512 = img_size
    task = 'multi',
    pred_obj_type= 'vessel',
    pretrained_path=None,
    train_path= '/home/ubuntu/giant/junang/TransUnet/shuffled_background_wire_v2_16666.h5+real_background_wire_v2_16666.h5+gear_background_wire_v2_16666.h5+shuffled_background_wire_v2_12000.h5+real_background_wire_v2_12000.h5+gear_background_wire_v2_12000.h5',
    test_path = None,

    workspace = '/home/junang/qubot/artifacts_removal/checkpoints/mixed_multi_task_v6',
)


config_doc['big'] = 'big model with higher (1024x1024) resolution '
config_defaults['big'] = Options(
    img_size=512,
    hidden_size=1024,
    attn_num_layers=24,
    attn_num_heads=16,
    mlp_dim=4096,
    resnet_pretrained_path = "../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz",
    pretrained_path =  '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz',


    wandb_name='1K_60_images_rayEmbedding_lr_pred_trained_1.2e-5', 
    resume = '/home/junang/qubot/checkpoints/best_model/model.safetensors',
    num_img=5000,

    batch_size=24,
    num_epochs=10,
    lr=1.2e-5,
    gradient_clip=1,
    gradient_accumulation_steps=1,
    mixed_precision='bf16'
)



AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)