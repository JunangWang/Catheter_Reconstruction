import torch
import yaml









def load_config(config_path):
    '''
    Load config
    '''
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) 

    return cfg

class ConeGeometry(object):
    """
    Cone beam CT geometry. Note that we convert to meter from millimeter. 1 m = 1000 mm
    """
    def __init__(self, data, device):

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data["DSD"]/1000 # Distance Source to Detector    (m) 
        self.DSO = data["DSO"]/1000  # Distance Source Origin        (m) 

        # Detector parameters
        self.nDetector = torch.tensor(data["nDetector"], device=device)  # number of pixels              (px)
        self.sPixel = torch.tensor(data["sPixel"],device=device)/1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.sPixel  # total size of the detector    (m)
        self.nPlane = (self.nDetector + 1) *3 - 1# total number of planes 
        
        # Image parameters
        self.nVoxel = torch.tensor(data["nVoxel"], device= device)  # number of voxels              (vx)
        self.sVoxel = torch.tensor(data["sVoxel"], device= device)/1000  # size of each voxel            (m)
        self.sVolume = self.nVoxel * self.sVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = torch.tensor(data["offOrigin"])/1000  # Offset of image from origin   (m)
        self.offDetector = torch.tensor(data["offDetector"])/1000  # Offset of Detector            (m)


def compute_pixels_p(geo, pose, opengl, device):

    x = torch.arange(geo.nDetector[0], device=device)
    y = torch.arange(geo.nDetector[1], device=device)
    X, Y = torch.meshgrid(x,y, indexing='xy')
    Z = torch.unsqueeze(torch.ones_like(X, device=device) * (geo.DSO-geo.DSD), dim=-1)
    pixels = torch.stack((X,Y),dim=-1)
    # pixel positions
    pixels_p = (pixels - geo.nDetector//2 + 0.5) * geo.sPixel
    if opengl:
        pixels_p[..., 0] *=-1
    print(pixels_p[0,0])
    pixels_p = torch.concatenate((pixels_p, Z), dim = -1) # (512, 512, 3)
    # transpose 
    
    pixels_p = pixels_p @ pose[:3, :3].T
    del x, y, X, Y, Z, pixels, pose
    torch.cuda.empty_cache()

    return pixels_p

def compute_min_max_intersection_for_rays(geo, rays, source_p, Plane_p, device):

    a = (Plane_p[[0,-1]] - source_p)/ torch.unsqueeze(rays, dim=2) # (512, 512, 2, 3)
    print(a.shape)
    print(a[0,0])
    a_min = torch.maximum(torch.max(torch.min(a, dim=2).values, dim=-1).values, torch.zeros((geo.nDetector[0], geo.nDetector[1]), device=device))
    a_min = a_min.reshape(-1,1)
    a_max = torch.minimum(torch.min(torch.max(a, dim=2).values, dim=-1).values, torch.ones((geo.nDetector[0], geo.nDetector[1]), device=device))
    a_max = a_max.reshape(-1,1)
    del a
    torch.cuda.empty_cache()
    return a_min, a_max


def render(config_path, pose, image, device = 'cuda', opengl = True):

    cfg = load_config(config_path)

    geo = ConeGeometry(cfg, device)

    # compute rays, a_s, source_p, Plane_p
    pose = pose.to(device=device)
    source_p = pose[:3, 3]

    # compute pixels_p
    pixels_p = compute_pixels_p(geo, pose, opengl, device=device)

    rays = pixels_p - source_p

    # planes
    Plane_x = torch.arange(geo.nVoxel[0] + 1, device=device)
    Plane_y = torch.arange(geo.nVoxel[1] + 1, device=device)
    Plane_z = torch.arange(geo.nVoxel[2] + 1, device=device)
    # Plane_Ys, Plane_Xs, Plane_Zs = np.meshgrid(Plane_z, Plane_y, Plane_x)
    Plane_p = (torch.stack((Plane_x, Plane_y, Plane_z), axis=-1) - geo.nVoxel//2)*geo.sVoxel
    print('Plane_p', Plane_p.shape)

    # compute a_min and a_max
    a_min, a_max = compute_min_max_intersection_for_rays(geo, rays, source_p, Plane_p, device)
    # (512*512, 1)
    # compute mask
    Plane_p_ext = Plane_p.reshape(1,1,geo.nDetector[0]+1,3) #(1,1, 513, 3)

    pixels_p_ext = torch.unsqueeze(pixels_p, dim= 2)
    # segment of rays a: 512,512 for rays, 513 plane
    a_s = (Plane_p_ext - source_p)/ (pixels_p_ext - source_p) # (512, 512, 513, 3)

    a_s = a_s.reshape(geo.nDetector.prod(), -1) # (512*512, 1539)
    a_s = torch.sort(a_s, dim=-1).values
    print('a_s', a_s.shape)
    print('a_min', a_min.shape)
    a_mask = (a_s > a_min) & (a_s < a_max)

    a_mask_min = torch.argmax(a_mask.int(), dim=1)
    # a_mask_max = a_mask.shape[1] - 1 - np.argmax(np.flip(a_mask, [1]), axis=1)  # 每行最后一个1的位置

    # set a_min and a_max to one
    a_mask[torch.arange(geo.nDetector.prod()), a_mask_min-1] = 1



    d = torch.linalg.norm(rays, dim=-1).reshape(-1, 1) # (512*512, 1)
    a_mid = torch.unsqueeze((a_s[:,1:] + a_s[:, :-1])/2, dim=-1) # (512*512, 1538, 1)

    # intersection length
    l = d*(a_s[:, 1:] - a_s[:, :-1]) # (512*512, 1538)

    idx = (source_p + a_mid*(rays.reshape(-1,1,3)) - Plane_p[0] )/geo.sVoxel
    idx = idx.int().reshape(-1,3) # (512*512*1538, 3)

    a_mask = a_mask[:,:-1] # (512*512, 1538)
    l= l*(a_mask)
    idx = idx*(a_mask.reshape(-1,1))
    # idx_masked = idx[a_mask_1.reshape(-1)]
    print('idx_masked shape', idx.shape)
    print(idx.min(), idx.max())

    del Plane_x, Plane_y, Plane_z, Plane_p, a_mid, d, a_s
    torch.cuda.empty_cache()
    image_rays = image[tuple(idx.T)].reshape(geo.nDetector.prod().item(), geo.nPlane[0]) #()
    projections = torch.sum(l*image_rays, dim=-1).reshape(geo.nDetector[0].item(),geo.nDetector[1].item())
    projections = projections.cpu().numpy()
    del a_mask, l, image_rays, idx
    torch.cuda.empty_cache()

    return projections