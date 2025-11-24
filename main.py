import os
import tyro
import math
import torch
import wandb
import logging
import numpy as np
from tqdm import tqdm
from torch.optim import Optimizer
from accelerate import Accelerator
from safetensors.torch import load_file
from core.dataset import DIASDataset
from core.options import AllConfigs
from core.models import TransUnet, TransUnet_MultiTask
from core.utils import rmae_loss, normalize_0_1, dice_loss, plot_points_and_lines_3d, plot_points_and_lines, load_ckpt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
import tifffile, cv2

from safetensors.torch import save_file
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():  
    # Fixing the random seed for reproducibility
    # seed = 42
    # random.seed(seed)  # Python's built-in random module
    # np.random.seed(seed)  # NumPy random generator
    # torch.manual_seed(seed)  # PyTorch CPU seed
    # torch.cuda.manual_seed_all(seed)  # PyTorch seed for all GPUs

    # Ensuring deterministic behavior
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  
    # torch.use_deterministic_algorithms(True)
    
    # Configration
    opt = tyro.cli(AllConfigs)
    os.makedirs(opt.workspace, exist_ok=True)
    os.makedirs(opt.workspace +'/images', exist_ok=True)
    os.makedirs(opt.workspace +'/images/iterations', exist_ok=True)
    os.makedirs(opt.workspace +'/images/epochs', exist_ok=True)
    os.makedirs(opt.workspace +'/images/test', exist_ok=True)
    wandb.init(project="artifacts_removal", name=opt.wandb_name)
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )
    logging.basicConfig(filename=opt.workspace+'output.log', level=logging.INFO, format='%(message)s')
    
    # model
    if opt.task == 'multi':
        model = TransUnet_MultiTask(opt)

    if opt.resnet_pretrained_path is not None:
        model.load_from(weights = np.load(opt.resnet_pretrained_path))
    
    if opt.pretrained_path is not None:
        model.load_from(weights = np.load(opt.pretrained_path))
    
    # fine-tune
    if opt.freeze_arts:
        print('---------freeze encoder and arts decode---------')
        model.freeze_encoder()
        model.freeze_arts_decode()
    
    
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')

        state_dict = model.state_dict()

        for k, v in ckpt.items():
            logging.info(f'{k} {v.shape}')

        load_ckpt(model, ckpt, opt.load_module_names)
        
        print("-----------------Weights loaded successfully-----------------")


    # dataset
    if opt.coord_dim == 2:
        from core.dataset import Wire2dDataset as WireDataset
    else:
        from core.dataset import Wire3dDataset as WireDataset
    if opt.task =='multi':
        # dataset = SynDataset(opt)
        if '+' in opt.train_path:
            train_dir = os.path.dirname(opt.train_path)
            train_path_list = os.path.basename(opt.train_path).split('+')
            dataset_list = []
            spatial_path = '/home/ubuntu/giant/guixiang/dataset_520/coords_50000.h5'
            for train_path in train_path_list:
                if '16666' in train_path:
                    train_dataset = WireDataset(opt, os.path.join(train_dir,train_path), num_img=opt.num_img//(len(train_path_list)), spatial_path=spatial_path)
                else:
                    train_dataset = WireDataset(opt, os.path.join(train_dir,train_path), num_img=opt.num_img//(len(train_path_list)))
                dataset_list.append(train_dataset)

            dataset = torch.utils.data.ConcatDataset(dataset_list)
        else:
            dataset = WireDataset(opt)
        # dataset = UnsupervisedDataset(opt)
    # split dataset to train and val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    if opt.test_path is not None:
        test_dataset = DIASDataset(opt)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=5e-2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)

    # scheduler
    if opt.freeze_arts:
        print('---------CosineAnnealingLR---------')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs*accelerator.num_processes)
    
    else:
        print('---------OneCycleLR---------')
        scheduler_total_steps = opt.num_epochs * len(train_dataloader)
        pct_start = 0.3
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=scheduler_total_steps, pct_start=pct_start)
    # scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, scheduler_total_steps, power=0.9, verbose=True)



    # accelerate
    if opt.test_path is not None:
        model, optimizer, train_dataloader, val_dataloader, test_dataloader,scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, test_dataloader,scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )

    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of parameters: ', num_params)
    # metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to(accelerator.device)
    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(accelerator.device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(accelerator.device) # need to noralize to [-1, 1]

    logging.info('Training Start')
    wandb.watch(model, log="all")
    best_metrics = -torch.inf
    for epoch in range(opt.num_epochs):
        # train
        model.train()

        for i, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}", ncols=100)):
            with accelerator.accumulate(model):

                # optimizer.zero_grad()

                out = model(data)
                loss = out['loss']
                accelerator.backward(loss)

                
                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                if opt.freeze_arts:
                    pass
                else:
                    scheduler.step()
                optimizer.zero_grad()

                step = epoch * len(train_dataloader) + i
                training_loss = loss.detach()
                training_arts_loss = out['arts_loss'].detach()
                training_seg_loss = out['seg_loss'].detach()
                training_spatial_loss = out['spatial_loss'].detach()
                training_rmse = out['rmse'].detach()
                training_MaxED = out['MaxED'].detach()
                training_MeanED = out['MeanED'].detach()
                training_MeanED_tip = out['MeanED_tip'].detach()

                training_arts_loss = accelerator.gather_for_metrics(training_arts_loss).mean()
                training_seg_loss = accelerator.gather_for_metrics(training_seg_loss).mean()
                training_spatial_loss = accelerator.gather_for_metrics(training_spatial_loss).mean()
                training_rmse = accelerator.gather_for_metrics(training_rmse).mean()
                training_MaxED = accelerator.gather_for_metrics(training_MaxED).mean()
                training_MeanED = accelerator.gather_for_metrics(training_MeanED).mean()
                training_MeanED_tip = accelerator.gather_for_metrics(training_MeanED_tip).mean()
                # elif opt.task =='seg':
                #     training_seg_loss = out['seg_loss'].detach()
                #     training_seg_loss = accelerator.gather_for_metrics(training_seg_loss).mean()
            # train loss
                training_loss = accelerator.gather_for_metrics(training_loss).mean()




            if accelerator.is_main_process:
                # logging
                if step % 1 == 0 and i != 0: 
                    lr = optimizer.state_dict()['param_groups'][0]['lr']

                    if opt.task =='multi':
                        message = f"\n[train] epoch: {epoch} iterations: {step} lr: {lr:.7f} " + \
                            f"training loss: {training_loss.item():.6f} " + \
                            f"training arts loss: {training_arts_loss.item():.6f} " + \
                            f"training seg loss: {training_seg_loss.item():.6f} " + \
                            f"training spatial loss: {training_spatial_loss.item():.6f} " + \
                            f"training rmse: {training_rmse.item():.6f} " + \
                            f"training MaxED: {training_MaxED.item():.6f} " + \
                            f"training MeanED: {training_MeanED.item():.6f} " + \
                            f"training MeanED_tip: {training_MeanED_tip.item():.6f} "
                        wandb.log({"step": step, 
                                "lr": lr, 
                                "training loss": training_loss.item(), 
                                "training arts loss": training_arts_loss.item(),
                                "training seg loss": training_seg_loss.item(),
                                "training spatial loss": training_spatial_loss.item(),
                                "training rmse": training_rmse.item(),
                                "training MaxED": training_MaxED.item(),
                                "training MeanED": training_MeanED.item(),
                                "training MeanED_tip": training_MeanED_tip.item(),
                                })

                    accelerator.print(message)
                    logging.info(message)
                if step%10000 == 0:
                    
                    for v in range(opt.num_view):
                        train_output = cv2.normalize(out['image'][0,v].detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        cv2.imwrite( f"{opt.workspace}/images/iterations/training_output_v{v}_{step}.png",train_output)

                        train_gt = cv2.normalize(data['gt'][0,v].detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        cv2.imwrite(f"{opt.workspace}/images/iterations/train_gt_v{v}_{step}.png", train_gt)

                    
                        train_mask = cv2.normalize(out['mask'][0,v].detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        cv2.imwrite( f"{opt.workspace}/images/iterations/training_mask_v{v}_{step}.png",train_mask)

                        if opt.pred_obj_type == 'wire':
                            pred_points = out['pred_3d'][0].detach().cpu().numpy()
                            pred_control_points = out['control_points'][0].detach().cpu().numpy()
                            gt_points = data['gt_3d'][0].reshape(opt.num_coords, opt.coord_dim).detach().cpu().numpy()
                            if opt.coord_dim == 3:
                                plot_points_and_lines_3d(pred_points, gt_points, pred_control_points, save_path=f"{opt.workspace}/images/iterations/training_wire_3d_{step}.png")
                            else:
                                plot_points_and_lines(pred_points, gt_points, pred_control_points, save_path=f"{opt.workspace}/images/iterations/training_wire_2d_{step}.png")

        # one epoch is finished
        if opt.freeze_arts:
            # old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
        # new_lr = optimizer.param_groups[0]['lr']
        # print(f'Learning rate updated: {old_lr} -> {new_lr}')
        # scheduler.step()

        # print(f'lr: {scheduler.get_last_lr()}')
            # accelerator.print(f'lr: {scheduler.get_last_lr()}')
        # else:
            # pass

        # checkpoint
        # accelerator.wait_for_everyone()
        # accelerator.save_model(model, opt.workspace)

        # eval
        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_arts_loss = 0
            val_seg_loss = 0
            val_spatial_loss = 0
            val_psnr = 0
            val_ssim = 0
            val_lpips = 0
            val_rmse = 0
            val_MaxED = 0
            val_MeanED = 0
            val_MeanED_tip = 0
            for i, data in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch}", ncols=100)):
                out = model(data)
                val_loss += out['loss'].detach() / len(val_dataloader)
                
                if opt.task == 'arts' or opt.task =='multi':
                    for v in range(opt.num_view):
                        val_psnr += psnr(normalize_0_1(out['image'][:,v:v+1]), normalize_0_1(data['gt'][:,v:v+1])) / len(val_dataloader) / opt.num_view
                        val_ssim += ssim(normalize_0_1(out['image'][:,v:v+1]), normalize_0_1(data['gt'][:,v:v+1])) / len(val_dataloader) / opt.num_view
                        val_lpips += lpips(normalize_0_1(out['image'][:,v:v+1]).repeat(1,3,1,1), normalize_0_1(data['gt'][:,v:v+1]).repeat(1,3,1,1)) / len(val_dataloader) / opt.num_view

                    
                    val_arts_loss += out['arts_loss'].detach() / len(val_dataloader)
                    val_seg_loss += out['seg_loss'].detach() / len(val_dataloader)
                    val_spatial_loss += out['spatial_loss'].detach() / len(val_dataloader)
                    val_rmse += out['rmse'].detach() / len(val_dataloader)
                    val_MaxED += out['MaxED'].detach() / len(val_dataloader)
                    val_MeanED += out['MeanED'].detach() / len(val_dataloader)
                    val_MeanED_tip += out['MeanED_tip'].detach() / len(val_dataloader)

            val_loss = accelerator.gather_for_metrics(val_loss).mean()
            val_arts_loss = accelerator.gather_for_metrics(val_arts_loss).mean()
            val_seg_loss  = accelerator.gather_for_metrics(val_seg_loss).mean()
            val_spatial_loss = accelerator.gather_for_metrics(val_spatial_loss).mean()
            val_rmse = accelerator.gather_for_metrics(val_rmse).mean()
            val_MaxED = accelerator.gather_for_metrics(val_MaxED).mean()
            val_MeanED = accelerator.gather_for_metrics(val_MeanED).mean()
            val_MeanED_tip = accelerator.gather_for_metrics(val_MeanED_tip).mean()
            val_psnr = accelerator.gather_for_metrics(val_psnr).mean()
            val_ssim = accelerator.gather_for_metrics(val_ssim).mean()
            val_lpips = accelerator.gather_for_metrics(val_lpips).mean()

            if accelerator.is_main_process:
                

                wandb.log({ "epoch": epoch,
                        "val loss": val_loss.item(), 
                        "val arts loss": val_arts_loss.item(),
                        'val seg loss': val_seg_loss.item(),
                        'val spatial loss': val_spatial_loss.item(),
                        'val rmse': val_rmse.item(),
                        'val MaxED': val_MaxED.item(),
                        "val psnr": val_psnr.item(),
                        "val ssim": val_ssim.item(),
                        "val lpips": val_lpips.item(),
                        "val MeanED": val_MeanED.item(),
                        "val MeanED_tip": val_MeanED_tip.item(),
                    })
                message = f"[eval] epoch: {epoch} val loss: {val_loss.item():.6f} " +  f"val arts loss: {val_arts_loss.item():.6f} "  + f"val seg loss: {val_seg_loss.item():.6f} "  + f"val spatial loss: {val_spatial_loss.item():.6f} " + f"val rmse: {val_rmse.item():.6f} " + f"val MaxED: {val_MaxED.item():.6f} " + f"val MeanED: {val_MeanED.item():.6f} " + f"val MeanED_tip: {val_MeanED_tip.item():.6f} " + f"val psnr: {val_psnr.item():.6f} " + f"val ssim: {val_ssim.item():.6f} " + f"val lpips: {val_lpips.item():.6f} "

                accelerator.print(message)
                logging.info(message)
            val_metrics = -val_rmse if opt.freeze_arts else val_ssim
            if val_metrics > best_metrics and accelerator.is_main_process:
                best_metrics = val_metrics
                # accelerator.wait_for_everyone()
                # accelerator.save_model(model, f"{opt.workspace}/best_model")
                state_dict = model.state_dict()
                os.makedirs(f"{opt.workspace}/best_model", exist_ok=True)
                save_file(state_dict, f"{opt.workspace}/best_model/model.safetensors")
                if opt.freeze_arts:
                    accelerator.print(f"Best rmse: {-best_metrics:.6f} saved to {opt.workspace}/best_model")
                else:
                    accelerator.print(f"Best ssim: {best_metrics:.6f} saved to {opt.workspace}/best_model")
    
            if accelerator.is_main_process:

                train_sample = next(iter(train_dataloader))
                train_out = model(train_sample)
                for v in range(opt.num_view):
                    train_output = (255*normalize_0_1(train_out['image'][:,v:v+1]).cpu().numpy()).astype(np.uint8)
                    cv2.imwrite(f"{opt.workspace}/images/epochs/training_output_v{v}_{epoch}.png", train_output[0,0])
                    train_gt = (255*normalize_0_1(train_sample['gt'][:,v:v+1]).cpu().numpy()).astype(np.uint8)
                    cv2.imwrite( f"{opt.workspace}/images/epochs/train_gt_v{v}_{epoch}.png", train_gt[0,0])
                    train_input = (255*normalize_0_1(train_sample['input'][:,v:v+1]).cpu().numpy()).astype(np.uint8)
                    cv2.imwrite( f"{opt.workspace}/images/epochs/train_input_v{v}_{epoch}.png", train_input[0,0])

                    train_mask = (255*train_out['mask'][:,v:v+1].cpu().numpy()).astype(np.uint8)
                    cv2.imwrite( f"{opt.workspace}/images/epochs/training_mask_v{v}_{epoch}.png",train_mask[0,0])
                
                if opt.pred_obj_type == 'wire':
                    pred_points = train_out['pred_3d'][0].detach().cpu().numpy()
                    pred_control_points = train_out['control_points'][0].detach().cpu().numpy()
                    # if control points in train_sample
                    if 'control_points' in train_sample:
                        gt_control_points = train_sample['control_points'][0].detach().cpu().numpy()
                    else:
                        gt_control_points = None
                    gt_points = train_sample['gt_3d'][0].reshape(opt.num_coords, opt.coord_dim).detach().cpu().numpy()
                    if opt.coord_dim == 3:
                        plot_points_and_lines_3d(pred_points, gt_points, pred_control_points, gt_control_points, save_path=f"{opt.workspace}/images/epochs/training_wire_3d_{epoch}.png")
                    else:
                        plot_points_and_lines(pred_points, gt_points, pred_control_points, save_path=f"{opt.workspace}/images/epochs/training_wire_2d_{epoch}.png")

                
                val_sample = next(iter(val_dataloader))
                out = model(val_sample)
                for v in range(opt.num_view):
                    val_output = (255*normalize_0_1(out['image'][:,v:v+1]).cpu().numpy()).astype(np.uint8)
                    cv2.imwrite( f"{opt.workspace}/images/epochs/validation_output_v{v}_{epoch}.png", val_output[0,0])
                    val_gt = (255*normalize_0_1(val_sample['gt'][:,v:v+1]).cpu().numpy()).astype(np.uint8)
                    cv2.imwrite( f"{opt.workspace}/images/epochs/val_gt_v{v}_{epoch}.png", val_gt[0,0])
                    val_input = (255*val_sample['input'][:,v:v+1].cpu().numpy()).astype(np.uint8)
                    cv2.imwrite( f"{opt.workspace}/images/epochs/val_input_v{v}_{epoch}.png", val_input[0,0])
                    val_mask = (255*out['mask'][:,v:v+1].cpu().numpy()).astype(np.uint8)
                    cv2.imwrite( f"{opt.workspace}/images/epochs/validation_mask_v{v}_{epoch}.png", val_mask[0,0])
                
                if opt.pred_obj_type == 'wire':
                    pred_points = out['pred_3d'][0].detach().cpu().numpy()
                    pred_control_points = out['control_points'][0].detach().cpu().numpy()
                    gt_points = val_sample['gt_3d'][0].reshape(opt.num_coords, opt.coord_dim).detach().cpu().numpy()
                    if opt.coord_dim == 3:
                        plot_points_and_lines_3d(pred_points, gt_points, pred_control_points, save_path=f"{opt.workspace}/images/epochs/validation_wire_3d_{epoch}.png")
                    else:
                        plot_points_and_lines(pred_points, gt_points, pred_control_points, save_path=f"{opt.workspace}/images/epochs/validation_wire_2d_{epoch}.png")

    if opt.test_path is not None:
    # inference on test dataset
        with torch.no_grad():
            model.eval()
            dice_loss_ = 0
            for i, data in enumerate(tqdm(test_dataloader, desc=f"Inference", ncols=100)):
                out = model(data)
                for v in range(opt.num_view):
                    test_output = out['image'][0,v].detach().cpu().numpy()
                    test_output = cv2.normalize(test_output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    test_input = (255*data['input'][0,v].detach().cpu().numpy()).astype(np.uint8)
                    cv2.imwrite(f"{opt.workspace}/images/test/test_input_v{v}_{i}.png", test_input)
                    cv2.imwrite(f"{opt.workspace}/images/test/test_output_v{v}_{i}.png", test_output)

                    test_mask = out['mask'][0,v]
                    test_mask_normal = cv2.normalize(test_mask.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    cv2.imwrite(f"{opt.workspace}/images/test/test_mask_v{v}_{i}.png", test_mask_normal)

                    if os.path.basename(opt.test_path) == 'images_DIAS.h5':
                        test_gt = data['gt'][0,v]

                        test_gt_normal = cv2.normalize(test_gt.detach().cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        cv2.imwrite(f"{opt.workspace}/images/test/test_gt_v{v}_{i}.png", test_gt_normal)

                        dice_loss_ += dice_loss(test_mask, test_gt).detach() / len(test_dataloader) / opt.num_view

            dice_loss_ = accelerator.gather_for_metrics(dice_loss_).mean()

            if accelerator.is_main_process:
                message = f"[Inference] F1: {1-dice_loss_.item():.6f} "
                accelerator.print(message)
                wandb.log(
                    {
                        "Test F1": 1-dice_loss_.item(),
                    }
                )


    logging.info('Training End')
    wandb.finish()


if __name__ == "__main__":
    main()
