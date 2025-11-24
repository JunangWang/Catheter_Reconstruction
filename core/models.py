import torch
import torch.nn as nn
import torch.nn.functional as F
from core.unet import UNet
from core.options_LGM import Options
from core.ViT import VisionTransformer, MultiTaskVisionTransformer
from core.utils import gradient_loss, gradient_2rd_loss, dice_loss, normalize_0_1, binorm_coeffs, generate_bspline_curve, batch_dice_loss_volume, bezier_curve
from core.Adn import ADN
from core.utils import normalize_std

class AdNet(nn.Module):
    def __init__(self, opt: Options):
        super(AdNet, self).__init__()
        self.opt = opt
        self.Adn = ADN(num_sides=3)
    
    def forward(self, data):
        result = {}
        if 'low' in data.keys():
            low = data['low']
            high = data['high'] # 0: without artifact, 1: artifact
        else:
            low = data['input']
            high = data['gt'] # 0: without artifact, 1: artifact

        # Adn
        # low -> low, high, train for encoder_art, encoder_low, decoder_art, decoder
        pred_ll, pred_lh = self.Adn.forward1(low)
        loss_ll = 20*F.l1_loss(pred_ll, low)
        loss_lh = 0*F.l1_loss(pred_lh, high) # set this to nonzero to train for paried data

        # low, high -> low, high, train for encoder_art, encoder_high, decoder_art, decoder
        pred_hl, pred_hh = self.Adn.forward2(low, high)
        loss_hh = 20*F.l1_loss(pred_hh, high)

        # low, high -> low, train for encode_art, encoder_high, decoder_art
        # lhl
        pred_lhl = self.Adn.forward_hl(pred_hl, pred_lh)
        loss_lhl = 10*F.l1_loss(pred_lhl, low)

        # low -> high, train for  encoder_low, decoder
        pred_hlh = self.Adn.forward_lh(pred_hl)
        loss_hlh = 20*F.l1_loss(pred_hlh, high)

        loss = loss_ll + loss_lh + loss_hh + loss_lhl + loss_hlh

        result['loss'] = loss
        result['pred_lh'] = pred_lh
        result['pred_hh'] = pred_hh
        result['pred_hl'] = pred_hl
        result['pred_hlh'] = pred_hlh
        result['pred_lhl'] = pred_lhl
        
        result['loss_ll'] = loss_ll
        result['loss_hh'] = loss_hh
        result['loss_hlh'] = loss_hlh
        result['loss_lhl'] = loss_lhl

        return result

        
        

#-------------------------------------------------------------
class TransUnet(nn.Module):
    def __init__(
            self,
            opt: Options,
    ):
        super(TransUnet, self).__init__()
        self.opt = opt
        self.ViT = VisionTransformer(
            img_size = opt.img_size,
            hidden_size = opt.hidden_size,
            patch_gird = opt.patch_gird,
            patch_size = opt.batch_size,
            resnet_num_layers = opt.resnet_num_layers,
            resnet_width_factor = opt.resnet_width_factor,
            in_channels = opt.input_channels,
            attn_num_layers = opt.attn_num_layers,
            attn_num_heads = opt.attn_num_heads,
            mlp_dim = opt.mlp_dim,
            qkv_bias = opt.qkv_bias,
            proj_bias = opt.proj_bias,
            embed_drop = opt.embed_drop,
            attn_drop = opt.attn_drop,
            attn_proj_drop = opt.attn_proj_drop,
            mlp_drop = opt.mlp_drop,
            decoder_channels = opt.decoder_channels,
            n_skip = opt.n_skip,
            skip_channels = opt.skip_channels,
            classifier = opt.classifier,
            vis = opt.vis
        )

    def load_from(self, weights):
        self.ViT.load_from(weights)


    def forward(self, data):
        
        result = {}
        x = data['input']
        gt = data['gt']
        origin_img_size = x.shape[-2:]
        # zoom inputs to img_size
        if self.opt.img_size!= origin_img_size[0] or self.opt.img_size!= origin_img_size[1]:
            x = F.interpolate(x, size=self.opt.img_size, mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=self.opt.img_size, mode='bilinear', align_corners=False)

       
        mask = (gt > (torch.amin(gt, dim=(1, 2, 3), keepdim=True) +0.015)).to(dtype=x.dtype, device=x.device)
        weight = torch.sum(mask, dim=(1, 2, 3), keepdim=True)
        
        weight =(self.opt.img_size**2 - weight) / (weight + 1e-4)
        # weight = 1
        weights = (weight*mask + 1).to(dtype=x.dtype, device=x.device)
        # pred arts or vessel
        if self.opt.pred_obj_type == 'arts':
            background = self.ViT(x) # range (-1,1)
            y = x - background
            gt_background = x - gt
            result['background'] = background
            mse_loss = F.l1_loss(weights*background, weights*gt_background)

            grad_loss = gradient_loss(weights*background, weights*gt_background)
            gradient_2_loss = gradient_2rd_loss(weights*background, weights*gt_background)
            background_loss = mse_loss + 0.5*grad_loss + 0.5*gradient_2_loss
        elif self.opt.pred_obj_type =='vessel':
            y = self.ViT(x)
            background_loss = 0

        # loss
       
        mse_loss = F.l1_loss(weights*y, weights*gt)

        grad_loss = gradient_loss(weights*y, weights*gt)
        gradient_2_loss = gradient_2rd_loss(weights*y, weights*gt)
        loss = mse_loss + 0.5*grad_loss + 0.5*gradient_2_loss + 10*background_loss

        result['mse_loss'] =mse_loss
        result['grad_loss'] = grad_loss
        result['grad_2rd_loss'] = gradient_2_loss
        result['loss'] = loss
        if self.opt.img_size!= origin_img_size[0] or self.opt.img_size!= origin_img_size[1]:
            result['output'] = F.interpolate(y, size=origin_img_size, mode='bilinear', align_corners=False)
        else:
            result['output'] = y
        return result

class TransUnet_MultiTask(TransUnet):
    def __init__(self, opt: Options):
        super(TransUnet_MultiTask, self).__init__(opt)
        self.opt = opt
        if opt.pred_obj_type == 'wire':
            if opt.coord_dim == 3:
                from core.ViT import MultiTaskVisionTransformer_3Dwire as MultiTaskVisionTransformer_wire
            elif opt.coord_dim == 2:
                from core.ViT import MultiTaskVisionTransformer_2Dwire as MultiTaskVisionTransformer_wire
            self.ViT = MultiTaskVisionTransformer_wire(
                img_size = opt.img_size,
                hidden_size = opt.hidden_size,
                patch_gird = opt.patch_gird,
                patch_size = opt.batch_size,
                resnet_num_layers = opt.resnet_num_layers,
                resnet_width_factor = opt.resnet_width_factor,
                in_channels = opt.input_channels,
                attn_num_layers = opt.attn_num_layers,
                attn_num_heads = opt.attn_num_heads,
                mlp_dim = opt.mlp_dim,
                qkv_bias = opt.qkv_bias,
                proj_bias = opt.proj_bias,
                embed_drop = opt.embed_drop,
                attn_drop = opt.attn_drop,
                attn_proj_drop = opt.attn_proj_drop,
                mlp_drop = opt.mlp_drop,
                decoder_channels = opt.decoder_channels,
                n_skip = opt.n_skip,
                skip_channels = opt.skip_channels,
                num_view = opt.num_view,
                coord_dim = opt.coord_dim,
                total_num_control_point = opt.total_num_control_point,
                classifier = opt.classifier,
                vis = opt.vis,
            )
            # self.binorm_coeffs = binorm_coeffs(self.opt.total_num_control_point-1)
            # self.N_sb_pints = self.opt.num_coords // self.opt.total_num_control_point
        elif opt.pred_obj_type == 'vessel':
            # TODO: add vessel model
            pass

    def load_from(self, weights):
        self.ViT.load_from(weights)

    def freeze_encoder(self):
        encoder = self.ViT.transformer
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        
    def freeze_arts_decode(self):
        decoder = self.ViT.decoder_arts
        decoder.eval()
        for param in decoder.parameters():
            param.requires_grad = False
        
        arts_head = self.ViT.arts_head
        arts_head.eval()
        for param in arts_head.parameters():
            param.requires_grad = False
        
        seg_head = self.ViT.seg_head
        seg_head.eval()
        for param in seg_head.parameters():
            param.requires_grad = False
            
            
    def freeze_head(self, head_name):

        if head_name == 'segmentation':
            head = self.ViT.seg_head
        elif head_name == 'artifacts_removal':
            head = self.ViT.arts_head
        else:
            raise ValueError('Invalid head name')

        head.eval() 
        for param in head.parameters():
            param.requires_grad = False
    
    def unfreeze_head(self, head_name):
        if head_name == 'segmentation':
            head = self.ViT.seg_head
        elif head_name == 'artifacts_removal':
            head = self.ViT.arts_head
        else:
            raise ValueError('Invalid head name')
        
        head.train()
        for param in head.parameters():
            param.requires_grad = True
            
    def forward(self, data):

        result = {}
        x = data['input']
        gt = data['gt']
        spatial_gt = data['gt_3d']

        batch_size = x.shape[0]
        origin_img_size = x.shape[-2:]
        # zoom inputs to img_size
        if self.opt.img_size!= origin_img_size[0] or self.opt.img_size!= origin_img_size[1]:
            x = F.interpolate(x, size=self.opt.img_size, mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=self.opt.img_size, mode='bilinear', align_corners=False)
        
        # if pred_obj_type is wire, then y, mask_logits, control_points
        # if pred_obj_type is vessel, then y, mask_logits, volume
        y, mask_logits, spatial_features = self.ViT(x)

        # segmentation task loss
        # dice loss
        gt_mask = (gt > (torch.amin(gt, dim=(2, 3), keepdim=True) +0.01)).to(dtype=x.dtype, device=x.device)
        weight = torch.sum(gt_mask, dim=(2, 3), keepdim=True)
        # print('weight shape: ', weight.shape)
        
        weight =(self.opt.img_size**2 - weight) / (weight + 1e-4)
        # weight = 1
        weights = (weight*gt_mask + 1).to(dtype=x.dtype, device=x.device)
        mask = F.sigmoid(mask_logits)
        dice_loss_ = dice_loss(mask, gt_mask)
        bce = nn.BCEWithLogitsLoss(pos_weight=weight)

        bce_loss = bce(mask_logits, gt_mask)
        seg_loss = 0.1*dice_loss_ + 0.1*bce_loss

        # arts task loss
        mse_loss = F.l1_loss(weights*y, weights*gt)

        grad_loss = gradient_loss(weights*y, weights*gt)
        gradient_2_loss = gradient_2rd_loss(weights*y, weights*gt)
        if self.opt.pred_obj_type == 'wire':
            arts_loss = (mse_loss + 0.5*grad_loss + 0.5*gradient_2_loss)

            # pred control points
            control_points = spatial_features.view(batch_size, -1, self.opt.coord_dim)
            # points = bezier_curve(control_points, self.binorm_coeffs, self.opt.num_coords) # (batch, n_points, coord_dim)
            points = generate_bspline_curve(control_points, self.opt.num_coords, spline_order=3)
            spatial_gt = spatial_gt.view(batch_size, -1, self.opt.coord_dim)
            control_points_gt = data['control_points']
            control_points_loss = F.mse_loss(control_points, control_points_gt)/5
            #--------------------------------
            # loss weight
            # loss_weight_front = torch.linspace(10, 1, self.opt.num_coords//2, device=x.device).view(1, -1)
            # loss_weight_back = torch.linspace(1, 10, self.opt.num_coords-self.opt.num_coords//2, device=x.device).view(1, -1)
            # loss_weight = torch.cat((loss_weight_front, loss_weight_back), dim=1)
            squared_dist = torch.sum((points - spatial_gt)**2, dim=2)

            loss_weight = torch.ones_like(squared_dist)
            loss_weight[:, -50:] *= 3
            points_mse_loss = (squared_dist*loss_weight).mean()
            # smooth loss
            pred_diff = points[:, 1:] - points[:, :-1]
            target_diff = spatial_gt[:, 1:] - spatial_gt[:, :-1]
            
            # direction loss
            pred_direction = F.normalize(pred_diff, dim=2)
            target_direction = F.normalize(target_diff, dim=2)
            direction_loss = (1.0 - torch.sum(pred_direction * target_direction, dim=2).mean() ) /50 
            # spatial_loss = points_mse_loss/10 + direction_loss
            #--------------------------------
 
            spatial_loss = 0*control_points_loss + 0*direction_loss + points_mse_loss/20
            # spatial_loss = 0*control_points_loss + 50*direction_loss + points_mse_loss*500


            result['pred_3d'] = points
            result['control_points'] = control_points

            with torch.no_grad():
                label_coords_b = spatial_gt*260 - 130
                pred_coords_b = points*260 - 130
                squared_loss = (label_coords_b-pred_coords_b)**2
                distance = torch.sqrt(torch.sum(squared_loss,dim=2)) # (batch, num_coords)
                rmse = torch.sqrt(torch.mean(squared_loss))
                result['MaxED'] = torch.max(distance)
                result['MeanED'] = torch.mean(distance)
                result['MeanED_tip'] = torch.mean(distance[:, -50:])
                result['rmse'] = rmse
        else:
            arts_loss = (mse_loss + 0.5*grad_loss + 0.5*gradient_2_loss)/weight.mean()
            
            # pred volume
            threshold = 0.7
            pred_volume = torch.sigmoid(spatial_features) 
            gt_volume = spatial_gt
            N_bg = torch.sum(gt_volume == 0)
            N_fg = torch.sum(gt_volume == 1)
            pos_weight = N_bg / N_fg
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            bce_volume = criterion(pred_volume, gt_volume)

            # dice loss
            dice_loss_volume = batch_dice_loss_volume(pred_volume, gt_volume).mean()

            # total loss
            spatial_loss =  0.5*bce_volume + 0.5*dice_loss_volume

            result['pred_3d'] = pred_volume > threshold




        # total loss
        loss = arts_loss + seg_loss + spatial_loss

        result['arts_loss'] = arts_loss
        result['seg_loss'] = seg_loss
        result['spatial_loss'] = spatial_loss
        result['loss'] = loss
        
        if self.opt.img_size!= origin_img_size[0] or self.opt.img_size!= origin_img_size[1]:
            result['image'] = F.interpolate(y, size=origin_img_size, mode='bilinear', align_corners=False)
            result['mask'] = F.interpolate(mask, size=origin_img_size, mode='bilinear', align_corners=False)

        else:
            result['image'] = y
            result['mask'] = mask
            
        return result



#-------------------------------------------------------------
class TransUnet_MultiTask_vessel(TransUnet):
    def __init__(self, opt: Options):
        super(TransUnet_MultiTask_vessel, self).__init__(opt)
        self.opt = opt
        self.ViT = MultiTaskVisionTransformer(
            img_size = opt.img_size,
            hidden_size = opt.hidden_size,
            patch_gird = opt.patch_gird,
            patch_size = opt.batch_size,
            resnet_num_layers = opt.resnet_num_layers,
            resnet_width_factor = opt.resnet_width_factor,
            in_channels = opt.input_channels,
            attn_num_layers = opt.attn_num_layers,
            attn_num_heads = opt.attn_num_heads,
            mlp_dim = opt.mlp_dim,
            qkv_bias = opt.qkv_bias,
            proj_bias = opt.proj_bias,
            embed_drop = opt.embed_drop,
            attn_drop = opt.attn_drop,
            attn_proj_drop = opt.attn_proj_drop,
            mlp_drop = opt.mlp_drop,
            decoder_channels = opt.decoder_channels,
            n_skip = opt.n_skip,
            skip_channels = opt.skip_channels,
            classifier = opt.classifier,
            vis = opt.vis
        )
    
    def load_from(self, weights):
        self.ViT.load_from(weights)
    def freeze_head(self, head_name):

        if head_name == 'segmentation':
            head = self.ViT.seg_head
        elif head_name == 'artifacts_removal':
            head = self.ViT.arts_head
        else:
            raise ValueError('Invalid head name')

        head.eval() 
        for param in head.parameters():
            param.requires_grad = False
    
    def unfreeze_head(self, head_name):
        if head_name == 'segmentation':
            head = self.ViT.seg_head
        elif head_name == 'artifacts_removal':
            head = self.ViT.arts_head
        else:
            raise ValueError('Invalid head name')
        
        head.train()
        for param in head.parameters():
            param.requires_grad = True
            
    def forward(self, data):

        result = {}
        x = data['input']
        gt = data['gt']
        origin_img_size = x.shape[-2:]
        # zoom inputs to img_size
        if self.opt.img_size!= origin_img_size[0] or self.opt.img_size!= origin_img_size[1]:
            x = F.interpolate(x, size=self.opt.img_size, mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=self.opt.img_size, mode='bilinear', align_corners=False)
        
        y, mask_logits = self.ViT(x)

        # segmentation task loss
        # dice loss
        gt_mask = (gt > (torch.amin(gt, dim=(1, 2, 3), keepdim=True) +0.01)).to(dtype=x.dtype, device=x.device)
        weight = torch.sum(gt_mask, dim=(1, 2, 3), keepdim=True)
        
        weight =(self.opt.img_size**2 - weight) / (weight + 1e-4)
        # weight = 1
        weights = (weight*gt_mask + 1).to(dtype=x.dtype, device=x.device)
        mask = F.sigmoid(mask_logits)
        dice_loss_ = dice_loss(mask, gt_mask)
        bce = nn.BCEWithLogitsLoss(pos_weight=weight)

        bce_loss = bce(mask_logits, gt_mask)
        seg_loss = 0.5*dice_loss_ + 0.5*bce_loss

        # arts task loss
        mse_loss = F.l1_loss(weights*y, weights*gt)

        grad_loss = gradient_loss(weights*y, weights*gt)
        gradient_2_loss = gradient_2rd_loss(weights*y, weights*gt)
        if self.opt.pred_obj_type == 'wire':
            arts_loss = (mse_loss + 0.5*grad_loss + 0.5*gradient_2_loss)
        else:
            arts_loss = (mse_loss + 0.5*grad_loss + 0.5*gradient_2_loss)/weight.mean()



        # total loss
        loss = arts_loss + seg_loss

        result['arts_loss'] = arts_loss
        result['seg_loss'] = seg_loss
        result['dice_loss'] = dice_loss_
        result['loss'] = loss
        if self.opt.img_size!= origin_img_size[0] or self.opt.img_size!= origin_img_size[1]:
            result['output'] = F.interpolate(y, size=origin_img_size, mode='bilinear', align_corners=False)
            result['mask'] = F.interpolate(mask, size=origin_img_size, mode='bilinear', align_corners=False)
        else:
            result['output'] = y
            result['mask'] = mask
        return result

class TransUnet_vessel_inference(TransUnet_MultiTask_vessel):
    def __init__(self, opt: Options):
        super(TransUnet_vessel_inference, self).__init__(opt)

    def forward(self, image):
        
        # downsample
        # print(image.dtype)
        # image = image.to(torch.float16)
        # image = F.interpolate(image, size=self.opt.img_size, mode='bilinear', align_corners=False).to(image.dtype)
        # image = normalize_0_1(image).to(image.dtype)
        # print(image.dtype)
        # print('input size-----',image.shape)
        y, mask_logits = self.ViT(image)
        mask = F.sigmoid(mask_logits)

        return y, mask
#-------------------------------------------------------------


class TransUnet_inference(TransUnet):
    def __init__(self, opt: Options):
        super(TransUnet_inference, self).__init__(opt)

    def forward(self, image):
        
        # downsample
        # print(image.dtype)
        image = image.to(torch.float16)
        image = F.interpolate(image, size=self.opt.img_size, mode='bilinear', align_corners=False).to(image.dtype)
        # print('input size-----',image.shape)
        y = self.ViT(image)

        return y

class TransUnet_MultiTask_inference(TransUnet_MultiTask):
    def __init__(self, opt: Options):
        super(TransUnet_MultiTask_inference, self).__init__(opt)

    def forward(self, image):

        # image shape (2, 512, 512), 2 for views
        # standardize the image
        image = normalize_std(image)
        
        # print(image.dtype)
        # image = image.to(torch.float16)
        # image = F.interpolate(image, size=self.opt.img_size, mode='bilinear', align_corners=False).to(image.dtype)
        # image = normalize_0_1(image).to(image.dtype)
        # print(image.dtype)
        # print('input size-----',image.shape)
        batch_size = image.shape[0]
        y, mask_logits, spatial_features = self.ViT(image)
        mask = F.sigmoid(mask_logits)

        control_points = spatial_features.view(batch_size, -1, self.opt.coord_dim)
        points = bezier_curve(control_points, self.binorm_coeffs, self.opt.num_coords)
        points = points*260 - 130

        return y, mask, points, control_points