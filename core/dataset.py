import os
import cv2
import torch
import random
import tifffile
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvF
from core.options import Options
from core.utils import get_rays, normalize_0_1, normalize_std
import h5py
from scipy.sparse import csr_matrix
from scipy.ndimage import zoom
from core.Bspline_utils import get_knots
from scipy import interpolate

class Wire3dDataset(Dataset):
    def __init__(self, opt: Options, train_path=None, num_img=None, spatial_path=None):
        '''
        The image of the dataset is normalized to [0, 1]
        '''
        self.opt = opt
        if spatial_path is not None:
            self.spatial_path = spatial_path
        else:
            self.spatial_path = opt.spatial_path


        if self.spatial_path is not None:
            with h5py.File(self.spatial_path, 'r') as f:
                dataset_name_list = list(f.keys())
                self.dataset_basename = dataset_name_list[0].split('_')[0]

        if train_path is None:
            self.h5_path = opt.train_path
        else:
            self.h5_path = train_path

        self.num_image_pairs = int(os.path.splitext(os.path.basename(self.h5_path).split('_')[-1])[0] )


        if num_img is not None:
            self.num_img = num_img
        else:
            self.num_img = self.opt.num_img
        if self.num_img > self.num_image_pairs:
            # warning
            print("WARNING: num_img should be smaller than or equal to num_image_pairs")
            self.num_img = self.num_image_pairs 
        
        self.spatial_idx_offset = 0
        if 'real' in self.h5_path:
            self.spatial_idx_offset = 0
            print('real', self.spatial_idx_offset)
        elif 'shuffled' in self.h5_path:
            self.spatial_idx_offset = self.num_image_pairs
            print('shuffled', self.spatial_idx_offset)
        elif 'gear' in self.h5_path:
            self.spatial_idx_offset = 2*self.num_image_pairs
            print('gear', self.spatial_idx_offset)
        self.knots = get_knots(0, 1, self.opt.total_num_control_point, 3)
    def __len__(self):
        return self.num_img

    def __getitem__(self, index):
        result = {}
        if self.spatial_path is None:
            with h5py.File(self.h5_path, 'r') as f:
                result['gt'] = f['gt'][index].astype(np.float32)
                result['input'] = f['input'][index].astype(np.float32)
                result['gt_3d'] = f['gt_3d'][index].astype(np.float32) # [1, 100, 3]
        else:
            with h5py.File(self.h5_path, 'r') as f:
                result['gt'] = f['gt'][index].astype(np.float32)
                result['input'] = f['input'][index].astype(np.float32)
            with h5py.File(self.spatial_path, 'r') as f:
                result['gt_3d'] = np.expand_dims(f[f'{self.dataset_basename}_{index + self.spatial_idx_offset}'][:].astype(np.float32), axis=0 )
            
        
        if np.random.rand() < 0.1:
            result['input'] = result['gt']
        result['input'] = torch.from_numpy(result['input'])
        result['gt'] = torch.from_numpy(result['gt'])
        result['gt_3d'] = (torch.from_numpy(result['gt_3d']) - -139) / (139 - -139)
        # print(result['gt_3d'][0][::5].shape)
        tck, u = interpolate.splprep(result['gt_3d'][0][::5].T, k=3, task=-1, t=self.knots)
        result['control_points'] = torch.from_numpy(np.array(tck[1]).T).to(torch.float32)


        

        # if result['input'].shape[-1] != self.opt.img_size or result['input'].shape[-2] != self.opt.img_size:
        #     result['input'] = torch.nn.functional.interpolate(result['input'].unsqueeze(0), size=self.opt.img_size, mode='bilinear', align_corners=True).squeeze(0)
        # if result['gt'].shape[-1] != self.opt.img_size or result['gt'].shape[-2] != self.opt.img_size:
        #     result['gt'] = torch.nn.functional.interpolate(result['gt'].unsqueeze(0), size=self.opt.img_size, mode='bilinear', align_corners=True).squeeze(0)


        result['input'] = normalize_std(result['input'])


        return result

class Wire2dDataset(Wire3dDataset):
    def __init__(self, opt: Options, train_path=None, num_img=None, spatial_path=None):
        super().__init__(opt, train_path, num_img, spatial_path)

    def __getitem__(self, index):
        result = {}
        if self.spatial_path is None:
            with h5py.File(self.h5_path, 'r') as f:
                result['gt'] = f['gt'][index,0:1].astype(np.float32)
                result['input'] = f['input'][index,0:1].astype(np.float32) #[1, 512, 512]
                result['gt_3d'] = f['gt_3d'][index,:,:,:2].astype(np.float32) # [1, 100, 2]
        else:
            with h5py.File(self.h5_path, 'r') as f:
                result['gt'] = f['gt'][index,0:1].astype(np.float32)
                result['input'] = f['input'][index,0:1].astype(np.float32) #[1, 512, 512]
            with h5py.File(self.spatial_path, 'r') as f:
                result['gt_3d'] = np.expand_dims(f[f'{self.dataset_basename}_{index + self.spatial_idx_offset}'][:].astype(np.float32), axis=0 )[:,:, :2]
            
        
        if np.random.rand() < 0.1:
            result['input'] = result['gt']
        result['input'] = torch.from_numpy(result['input'])
        result['gt'] = torch.from_numpy(result['gt'])
        result['gt_3d'] = (torch.from_numpy(result['gt_3d']) - -139) / (139 - -139)
        # tck, u = interpolate.splprep(result['gt_3d'][0][::10].T, k=3, task=-1, t=self.knots)
        result['control_points'] = torch.zeros((self.opt.total_num_control_point, self.opt.coord_dim)).to(torch.float32)




        result['input'] = normalize_std(result['input'])
        return result
#-------------------------------------------------------------------
class UnsupervisedDataset(Dataset):
    def __init__(self, opt: Options):
        '''
        The shuffled dataset is normalized to [0, 1]
        '''
        self.opt = opt
        self.h5_path = opt.train_path
        
        real_h5_path = '/home/ubuntu/giant/junang/TransUnet/Unsupervised_train_DIAS.h5'
        with h5py.File(real_h5_path, 'r') as f:
            self.real_DSA = f['image'][:]

        self.num_image_pairs = int(os.path.splitext(os.path.basename(self.h5_path).split('_')[-1])[0] )
        if self.opt.num_img > self.num_image_pairs:
            # warning
            print("WARNING: num_img should be smaller than or equal to num_image_pairs")
            self.opt.num_img = self.num_image_pairs 
        

    

    def __len__(self):
        return self.opt.num_img

    def __getitem__(self, index):
        result = {}
        if index < self.real_DSA.shape[0]:
            result['low'] = 1- self.real_DSA[index,0:1].astype(np.float16)
        
            x_min = result['low'].min()
            x_max = result['low'].max()
            result['low'] = (result['low'] - x_min) / (x_max - x_min + 1e-10)

            with h5py.File(self.h5_path, 'r') as f:
                result['high'] = f['gt'][index].astype(np.float16)

        else:
            with h5py.File(self.h5_path, 'r') as f:
                result['low'] = f['input'][index].astype(np.float16)
                result['high'] = f['gt'][index].astype(np.float16)
        result['low'] = torch.from_numpy(result['low'])
        result['high'] = torch.from_numpy(result['high'])

        result['low'] = torch.nn.functional.interpolate(result['low'].unsqueeze(0), size=self.opt.img_size, mode='bilinear', align_corners=True).squeeze(0)

        result['high'] = torch.nn.functional.interpolate(result['high'].unsqueeze(0), size=self.opt.img_size, mode='bilinear', align_corners=True).squeeze(0)
        return result


class DIASDataset(Dataset):
    def __init__(self, opt: Options):

        self.opt = opt

        if opt.test_path is not None:
            self.h5_path = opt.test_path
        else:
            self.h5_path = opt.train_path 

        with h5py.File(self.h5_path, 'r') as h5file:
            self.images = h5file['image'][:]
            # if these is 'label' in h5file dataset
            if 'label' in h5file:
                self.labels = h5file['label'][:]
            else:
                self.labels = np.zeros_like(self.images)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        result = {}

        result['input'] = 1- self.images[idx,0:2].astype(np.float32)
        result['gt'] =  self.labels[idx,0:2].astype(np.float32)
        result['gt_3d'] = np.zeros((1, self.opt.num_coords, self.opt.coord_dim))
    
        x_mean = np.mean(result['input'])
        x_std = np.std(result['input'])
        y_min = result['gt'].min()
        y_max = result['gt'].max()
        result['input'] = (result['input'] - x_mean) / (x_std + 1e-10)
        

        result['gt'] = (result['gt'] - y_min) / (y_max - y_min + 1e-10)
        # print('input shape', result['input'].dtype, 'gt shape', result['gt'].dtype)

        return result




















class ObjaverseDataset(Dataset):

    def __init__(self, opt: Options, samples: list):
        
        self.opt = opt
        self.items = []
        for i in samples:
            selected_image_groups = random.sample(range(60), self.opt.num_image_groups)
            for j in range(len(selected_image_groups)):
                image_index = selected_image_groups[j]
                chosen_views = [i, image_index]
                self.items.append(chosen_views)
                
        if self.opt.ray_embedding:
            image_folder = self.opt.train_path + 'images/'
            self.rayDirection = np.load(image_folder + 'rayDirection.npy')
            self.rayOrigin = np.load(image_folder + 'rayOrigin.npy')
            self.sourcePoseArray = np.load(image_folder + 'sourcePoseMatrix.npy')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        results = {}
        vascular_index = self.items[idx][0]
        view_index = self.items[idx][1]

        # load input images w/wo ray embeddings
        start = int(vascular_index//2000*2000)
        with h5py.File(self.opt.train_path + 'dataset_image' + '_' + str(start) + '.h5', 'r') as f:
            images = f['images'][(vascular_index-start)*60+view_index] 
        # images [0, 1]

        # images = 0 - np.log(images)

        # flip the image left and right
        # either image flip or ray direction flip
        # images = np.flip(images, axis=-1)

        # # normalize image
        # images = (images-images.min()) / (images.max()-images.min())

        images = torch.from_numpy(images.astype(np.float16)).contiguous().unsqueeze(1) # (V, 1, H, W)

        source_poses = []

        if self.opt.ray_embedding:

            vids = [view_index, view_index+60, view_index+120]
            for i in range(len(vids)):
                image = images[i]
                # load source pose martices and normalize them
                source_pose = self.sourcePoseArray[vids[i]]
                c2w = torch.tensor(source_pose, dtype=torch.float32).contiguous()
                radius = torch.norm(c2w[:3,3].contiguous())
                c2w[:3,3] *= 1/radius
                source_poses.append(c2w)


        source_poses = torch.stack(source_poses, dim=0) # [V, 4, 4]
        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(source_poses[0].contiguous())
        source_poses = transform.unsqueeze(0) @ source_poses  # [V, 4, 4]

        if self.opt.ray_embedding:
            rays_embeddings = []
            for i in range(len(vids)):
                # rays_o, rays_d = get_rays(source_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
                # one can directly load rays_o and rays_d from rayDirection and rayOrigin matrix
                # either image flip or ray direction flip
                rays_d = torch.from_numpy(np.flip(self.rayDirection[vids[i]], axis=1).copy()).to(torch.float32)
                rays_o = torch.from_numpy(self.rayOrigin[vids[i]]).expand_as(rays_d).to(torch.float32)
                rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings.append(rays_plucker)
            rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            # images = torch.stack(images, dim=0) # [V, 1, H, W]
            inputs = torch.cat([images, rays_embeddings], dim=1) # [V=3, 7, H, W]
            results['input'] = inputs
        else:
            # inputs = torch.stack(inputs, dim=0) # [V, 1, H, W]
            inputs = images
            results['input'] = inputs

        # load ground truth volumes
        start_v = int(vascular_index//10000*10000)
        with h5py.File(self.opt.train_path + 'dataset_sparse_volume'+ '_' + str(start_v) + '.h5', "r") as f:
            group = f[f"matrix_{vascular_index}"]
            data = group["data"][:]
            indices = group["indices"][:]
            indptr = group["indptr"][:]
            shape = tuple(group["shape"][:])
            sparse_matrix = csr_matrix((data, indices, indptr), shape=shape)
        image_size = self.opt.output_size
        dense_matrix = sparse_matrix.toarray()
        volume = dense_matrix.reshape((image_size,image_size,image_size))
        volume = torch.from_numpy(volume).contiguous()
        results['output'] = volume 

        return results