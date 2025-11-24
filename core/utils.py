import numpy as np
import tifffile
import torch
from kiui.op import safe_normalize
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
def save_binary_voxel_to_tiff(voxel_tensor, output_path):
    
    voxel_tensor = voxel_tensor.cpu()
    
    voxel_numpy = voxel_tensor.detach().numpy()

    voxel_numpy = (voxel_numpy ).astype(np.bool_)

    # save tiff file
    tifffile.imwrite(
        output_path,
        voxel_numpy,
        compression='zlib',
        photometric='minisblack'
    )

def get_rays(pose, h, w, fovy, opengl=True):

    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy)) # in pixel unit

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal, # unit 1
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]
    # camera_dirs is the direction in source local coords
    rays_d = camera_dirs @ pose[:3, :3].contiguous().transpose(0, 1).contiguous()  # [hw, 3]

    # rays_d[:,-1] *= -1 # convert to Xray source to detector 
    rays_o = pose[:3, 3].contiguous().unsqueeze(0).expand_as(rays_d) # [hw, 3] 

    rays_o = rays_o.contiguous().view(h, w, 3)
    rays_d = safe_normalize(rays_d).contiguous().view(h, w, 3)

    return rays_o, rays_d


def rmae_loss(predictions, targets, epsilon=1e-8):
    # 计算相对误差：|y_pred - y_true| / (|y_true| + epsilon)
    relative_error = torch.abs(predictions - targets) / (torch.abs(targets) + epsilon)
    # 计算均值：mean(relative_error)
    rmae = torch.mean(relative_error)
    return rmae

def normalize(x):
    '''
    normalize x (batch, channels, height, width) to [-1,1]
    '''
    x_max = torch.amax(x, dim=(1,2,3), keepdim=True)
    x_min = torch.amin(x, dim=(1,2,3), keepdim=True)

    x = (x - x_min) / (x_max - x_min + 1e-8)
    x = 2 * x - 1
    return x
def normalize_std(x):
    '''
    standardization normalize x (batch, channels, height, width) to [-1,1]
    '''
    if len(x.shape) == 4:
        x_mean = torch.mean(x, dim=(1,2,3), keepdim=True)
        x_std = torch.std(x, dim=(1,2,3), keepdim=True)
        x = (x - x_mean) / (x_std + 1e-8)
    else:
        x_mean = torch.mean(x)
        x_std = torch.std(x)
        x = (x - x_mean) / (x_std + 1e-8)
    return x
def normalize_0_1(x):
    '''
    normalize x (batch, channels, height, width) to [0,1]
    '''
    if len(x.shape) == 4:

        x_max = torch.amax(x, dim=(1,2,3), keepdim=True)
        x_min = torch.amin(x, dim=(1,2,3), keepdim=True)
    else:
        x_max = torch.max(x)
        x_min = torch.min(x)


    x = (x - x_min) / (x_max - x_min + 1e-4)

    # torch.clip(x, 0, 1, out=x)
    return x

def gradient_loss(image1, image2):
    # 水平和垂直方向的梯度
    h_gradient1 = image1[:, :, :, :-1] - image1[:, :, :, 1:]
    v_gradient1 = image1[:, :, :-1, :] - image1[:, :, 1:, :]
    
    h_gradient2 = image2[:, :, :, :-1] - image2[:, :, :, 1:]
    v_gradient2 = image2[:, :, :-1, :] - image2[:, :, 1:, :]

    # print('is nan y', torch.isnan(h_gradient1).any().item())  # 检查预测值中是否有 NaN
    # print('is nan gt', torch.isnan(h_gradient2).any().item())  # 检查标签中是否有 NaN

    # print('is inf y', torch.isinf(v_gradient1).any().item())  # 检查预测值中是否有 Inf
    # print('is inf gt',torch.isinf(v_gradient2).any().item())  # 检查标签中是否有 Inf

    # print('is h1-h2 nan', torch.isnan(torch.mean(torch.abs(h_gradient1- h_gradient2))).any().item())
    # print('is v1-v2 nan', torch.isnan(torch.mean(torch.abs(v_gradient1-v_gradient2))).any().item())

    return torch.mean(torch.abs(h_gradient1- h_gradient2)) + torch.mean(torch.abs(v_gradient1-v_gradient2))

def gradient_2rd_loss(preds, gt):

    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=preds.device, dtype=preds.dtype)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    lap_preds = F.conv2d(preds, kernel.repeat(preds.shape[1], 1, 1, 1), padding=1, groups=preds.shape[1])
    lap_gt = F.conv2d(gt, kernel.repeat(gt.shape[1], 1, 1, 1), padding=1, groups=gt.shape[1])
    loss = torch.mean(torch.abs(lap_preds - lap_gt))

    return loss

def load_ckpt(model, ckpt, load_module_names=None):

    state_dict = model.state_dict()

    def load_model_parameters(k, v, state_dict):
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f'[WARN] mismatching shape for para {k}: checkpoint {v.shape}!=model{state_dict[k].shape}, ignore')
        else:
            print(f'[WARN] unexpected param {k}: {v.shape}')

    for k, v in ckpt.items():
        k = k.replace('module.', '')
        # load all model parameters if load_model_name is None
        if load_module_names is None:
            load_model_parameters(k, v, state_dict)
        else:
            for model_name in load_module_names:
                if model_name in k:
                    print(f'[INFO] loading {k} from {model_name}...')
                    load_model_parameters(k, v, state_dict)
                    break
    return model


def dice_loss(pred, target):

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice_score = (2. * intersection + 1e-8) / (union + 1e-8)

    return 1 - dice_score

def batch_dice_loss_volume(pred, target):
    '''
    pred: (batch, h, w, d)
    target: (batch, h, w, d)
    return: (batch)
    '''
    # flatten (batch, h, w, d) to (batch, h*w*d)
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice_score = (2. * intersection + 1e-8) / (union + 1e-8)

    return 1 - dice_score
    
    
def plot_points_and_lines(pred_points, gt_points, pred_control_points, image_size=(300, 300), save_path=None):
    # 创建白色背景图像
    img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

    # 将points转换为整数坐标
    gt_points_max = np.max(gt_points, axis=0, keepdims=True)
    gt_points_min = np.min(gt_points, axis=0, keepdims=True)
    gt_points_normalized = (gt_points - gt_points_min) / (gt_points_max - gt_points_min)
    pred_points_normalized = (pred_points - gt_points_min) / (gt_points_max - gt_points_min)
    pred_control_points_normalized = (pred_control_points - gt_points_min) / (gt_points_max - gt_points_min)
    pred_points = np.array(pred_points_normalized*260+20, dtype=np.int32)
    pred_control_points = np.array(pred_control_points_normalized*260+20, dtype=np.int32)
    gt_points = np.array(gt_points_normalized*260+20, dtype=np.int32)

    # 绘制线条
    cv2.polylines(img, [pred_points], isClosed=False, color=(0,0,255), thickness=2)
    cv2.polylines(img, [gt_points], isClosed=False, color=(255,0,0), thickness=2)
    for i in range(pred_control_points.shape[0]):
        cv2.circle(img, pred_control_points[i], 3, (0,255,0), -1)

    cv2.imwrite(save_path, img)

def plot_points_and_lines_3d(pred_points, gt_points, pred_control_points, gt_control_points=None, save_path=None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pred_points[:,0], pred_points[:,1], pred_points[:,2], c='r', linewidth=2)
    ax.plot(gt_points[:,0], gt_points[:,1], gt_points[:,2], c='b', linewidth=2)

    ax.scatter(pred_control_points[:,0], pred_control_points[:,1], pred_control_points[:,2], c='g', marker='o')
    if gt_control_points is not None:
        ax.scatter(gt_control_points[:,0], gt_control_points[:,1], gt_control_points[:,2], c='y', marker='o')
        gt_points_Bspline = generate_bspline_curve(torch.from_numpy(np.expand_dims(gt_control_points, axis=0)).float(), n_points=1000)
        ax.plot(gt_points_Bspline[0,:,0], gt_points_Bspline[0,:,1], gt_points_Bspline[0,:,2], c='y', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(save_path)
    plt.close()
def binorm_coeffs(n, device='cuda'):
    """
    Returns the nth binomial coefficients.
    """
    k = torch.arange(n+1, device=device)

    # 使用 torch.lgamma 计算对数阶乘
    # C(n,k) = n! / (k! * (n-k)!)
    # log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
    log_factorial_n = torch.lgamma(torch.tensor(n + 1, device=device).float())
    log_factorial_k = torch.lgamma(k + 1)
    log_factorial_n_minus_k = torch.lgamma(torch.tensor(n + 1, device=device).float() - k)

    # 计算对数二项式系数
    log_coeffs = log_factorial_n - log_factorial_k - log_factorial_n_minus_k

    # 转换回普通尺度
    coeffs = torch.exp(log_coeffs)

    return coeffs

def bezier_curve(control_points, coeffs, num_points):
    """
    Returns the point on a bezier curve defined by the given points in the range [0,1] at time t.
    control_points: tensor of shape (batch_size, num_segments, num_control_points, 2)
    coeffs: tensor of shape (num_control_points)

    """
    # single bezier control point
    sb_control_point = coeffs.shape[0]

    # control_points = control_points.unfold(1, sb_control_point, sb_control_point-1).permute(0, 1, 3, 2) .contiguous() #(batch_size, segments, sb_control_points,  2)
    if len(control_points.shape) == 3:
        control_points = control_points.unsqueeze(1)
    batch_size, num_segments, sb_control_point, dim = control_points.shape
    
    num_sb_points = num_points // num_segments

    k = torch.arange(sb_control_point, device=control_points.device).float()
    k = k.view(1, 1, 1, sb_control_point) # (1, 1, 1, sb_control_point)
    t = torch.linspace(0, 1, num_sb_points, device=control_points.device)
    t = t.view(1, 1, num_sb_points, 1) # (1, 1, num_sb_points, 1)
    coeffs = coeffs.view(1, 1, 1, sb_control_point) # (1, 1, 1, sb_control_points)


    # compute Bernstein base function
    # coeffs*t^k*(1-t)^(n-k)
    t_powers = torch.pow(t, k) # (1, 1, num_sb_points, sb_control_points)
    one_minus_t_powers = torch.pow(1 - t, sb_control_point - k-1) # (1, 1, num_sb_points, sb_control_points)
    bernstein_base = coeffs * t_powers * one_minus_t_powers # (1, 1, num_sb_points, sb_control_points)

    # compute bezier curve
    bernstein_base = bernstein_base.unsqueeze(-1) # (1, 1, num_sb_points, sb_control_points, 1)
    control_points = control_points.unsqueeze(2) # (batch_size, segments, 1 sb_control_points, 2)
    # print(bernstein_base.shape, control_points.shape)

    bezier_curve = torch.sum(bernstein_base * control_points, dim=3) # (batch_size, segments,num_sb_points, 2)

    bezier_curve = bezier_curve.reshape(batch_size, num_points, dim) # (batch_size, num_points, 2)


    return bezier_curve

def independent_control_points(control_points, order_bezier):
    '''
    control_points: [B, num_total_control_points, D]
    order_bezier: int
    '''
    sb_control_point = order_bezier + 1
    control_points = control_points.unfold(1, sb_control_point, sb_control_point-1).permute(0, 1, 3, 2) .contiguous() #(batch_size, segments, sb_control_points,  2)
    mu_c1 = torch.linalg.norm(control_points[:, :-1, -1]- control_points[:, :-1, -2], dim=-1)/(torch.linalg.norm((control_points[:, 1:, 1] - control_points[:, 1:, 0]), dim=-1) + 1e-8)

    mu_c2 = torch.linalg.norm(control_points[:, 1:, -1]- 2*control_points[:, :-1, -2] + control_points[:, :-1, -3], dim=-1)/(torch.linalg.norm((control_points[:, 1:, 2] - 2*control_points[:, 1:, 1] + control_points[:, 1:, 0]), dim=-1) + 1e-8)
    # every segment 2rd control point is dependent on the previous segment's 3rd and 4th control points
    # now segement 1th point = 2* prev_4th_point - prev_3rd_point
    control_points[:, 1:, 1] = mu_c1.unsqueeze(-1)*(control_points[:, :-1, -1] - control_points[:, :-1, -2]) + control_points[:, 1:, 0]
    # every segment 3rd control point is dependent on the previous segment's last and 2rd last control points and current 2rd control point
    control_points[:, 1:, 2] = mu_c2.unsqueeze(-1)*(control_points[:, 1:, -1]- 2*control_points[:, :-1, -2] + control_points[:, :-1, -3]) + 2*control_points[:, 1:, 1] - control_points[:, 1:, 0]

    return control_points

def independent_control_points2full(indep_control_points, order_bezier, N_segments):
    '''
    control_points: [B, num_total_control_points, D]
    order_bezier: int
    '''
    sb_control_point = order_bezier + 1
    batch_size, num_total_indep_control_points, dim = indep_control_points.shape
    control_points = torch.zeros((batch_size, N_segments, sb_control_point, dim), device=indep_control_points.device, dtype=indep_control_points.dtype)

    control_points[:, 0, :] += indep_control_points[:, :sb_control_point]
    # since c0, c1, c2 consistity, the first three control points are dependent on the previous segment's last and 2rd last control points and current 2rd control point
    control_points[:,1:, 3:] += indep_control_points[:, sb_control_point:].reshape(batch_size, N_segments-1, -1, dim)
    # c0
    control_points[:, 1:, 0] = control_points[:, :-1, -1]

    for i in range(1, N_segments):
        # every segment 2rd control point is dependent on the previous segment's 3rd and 4th control points
        # now segement 1th point = 2* prev_4th_point - prev_3rd_point
        control_points[:, i, 1] = 2*control_points[:, i-1, -1] - control_points[:, i-1, -2]
        # every segment 3rd control point is dependent on the previous segment's last and 2rd last control points and current 2rd control point
        control_points[:, i, 2] = 2*control_points[:, i, 1] - 2*control_points[:, i-1, -2] + control_points[:, i-1, -3]

    return control_points

def chamfer_distance(x, y):
    """
    计算两组点集之间的Chamfer距离
    x: 预测点集 [batch_size, n_points, dim]
    y: GT点集 [batch_size, m_points, dim]
    """
    # 计算x中每个点到y中最近点的距离
    x_to_y = torch.min(torch.sum((x.unsqueeze(2) - y.unsqueeze(1))**2, dim=3), dim=2)[0]
    # 计算y中每个点到x中最近点的距离
    y_to_x = torch.min(torch.sum((y.unsqueeze(2) - x.unsqueeze(1))**2, dim=3), dim=2)[0]
    
    # 双向距离之和
    chamfer_loss = torch.mean(x_to_y) + torch.mean(y_to_x)
    return chamfer_loss

def continuity_loss(control_points, order_bezier):
    """
    确保相邻Bezier曲线片段之间的连续性
    pred_curves: [B,  segments, sb_control_points, D] - 预测的Bezier曲线控制点
    """
    # single bezier curve control point 
    sb_control_point = order_bezier + 1
    # control_points = control_points.unfold(1, sb_control_point, sb_control_point-1).permute(0, 1, 3, 2) .contiguous() # [B, N_segments, 4, D]
    # 可以进一步添加切线连续性约束
    # 前一段的最后两个控制点与后一段的前两个控制点应该共线
    tangent_pred_1 = control_points[:, :-1, -1] - control_points[:, :-1, -2]  # [B, N_segments-1, D]
    tangent_pred_2 = control_points[:, 1:, 1] - control_points[:, 1:, 0]  # [B, N_segments-1, D]
    
    # 归一化切线向量
    tangent_pred_1 = tangent_pred_1 / (torch.norm(tangent_pred_1, dim=2, keepdim=True) + 1e-10)
    tangent_pred_2 = tangent_pred_2 / (torch.norm(tangent_pred_2, dim=2, keepdim=True) + 1e-10)
    
    # 计算切线连续性损失 - 方向应该一致
    tangent_consistency = torch.mean(
        1 - torch.nn.functional.cosine_similarity(tangent_pred_1, tangent_pred_2, dim=2)
    )
    
    # 总连续性损失
    loss =  tangent_consistency
    return loss

from core.BSpline import BSpline  # 假设上面的代码保存在BSpline.py文件中

def generate_bspline_curve(control_points, n_points=100, spline_order=3):
    """
    根据控制点生成B样条曲线上的离散点
    
    参数:
        control_points: np.array, 形状为(n, d)的控制点数组，n是控制点数量，d是维度
        n_points: int, 要生成的离散点数量
        spline_order: int, B样条阶数
    
    返回:
        np.array, 形状为(n_points, d)的曲线上的离散点
    """
    # 获取控制点数量和维度
    n_control_points = control_points.shape[1]
    
    # 创建B样条对象
    # 注意：n_bases应该等于控制点数量
    bspline = BSpline(start=0, end=1, n_bases=n_control_points, spline_order=spline_order)
    
    # 生成参数空间中的均匀分布点
    t = torch.linspace(0, 1, n_points, device=control_points.device)
    
    # 计算每个参数值处的基函数值
    basis_functions = bspline.predict(t).unsqueeze(0).repeat(control_points.shape[0], 1, 1)
    
    
    # 计算曲线上的点
    # for i in range(dim):
        # 对每个维度，曲线上的点是控制点和基函数的线性组合
        # curve_points[:, i] = basis_functions @ control_points[:, i]
    
    curve_points = torch.bmm(basis_functions.to(torch.float64), control_points.to(torch.float64))
    return curve_points
