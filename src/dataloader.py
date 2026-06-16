import pandas as pd
import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
from itertools import product
from itertools import combinations
from matplotlib import pyplot as plt 
import numpy as np
import itertools
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy.ndimage import center_of_mass

def resize_volume_nifti(img, target_shape=(128,128,128)):

    factors = [t/s for t, s in zip(target_shape, img.shape)]
    resized = zoom(img, factors, order=3)  # cubic interpolation

    return resized

def crop_around_mask_center(image, mask, crop_size=(128, 128, 128)):
    # Convert to NumPy if needed
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image

    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask
    # Get center of mass of mask
    com = center_of_mass(mask_np)  # returns (z, y, x)
    com = np.round(com).astype(int)
    # Calculate crop bounds
    d, h, w = image_np.shape[-3:]
    cd, ch, cw = crop_size
    cz, cy, cx = com

    z_min = max(cz - cd // 2, 0)
    y_min = max(cy - ch // 2, 0)
    x_min = max(cx - cw // 2, 0)

    z_max = min(z_min + cd, d)
    y_max = min(y_min + ch, h)
    x_max = min(x_min + cw, w)

    # Adjust if crop goes out of bounds
    z_min = z_max - cd
    y_min = y_max - ch
    x_min = x_max - cw

    # Crop image
    if image_np.ndim == 3:
        cropped = image_np[z_min:z_max, y_min:y_max, x_min:x_max]
    elif image_np.ndim == 4:
        cropped = image_np[:, z_min:z_max, y_min:y_max, x_min:x_max]
    else:
        raise ValueError("Image must be 3D or 4D (C, D, H, W)")

    # Convert back to Tensor if input was Tensor
    if isinstance(image, torch.Tensor):
        cropped = torch.from_numpy(cropped).to(image.device).type(image.dtype)

    return cropped

def norm_images(I):
    I = (I - I.min())/ (I.max()-I.min()+1e-8)
    return I

def center_crop_3d(tensor, crop_size=(128, 128, 128)):
    """
    Center crop a 3D tensor to crop_size.

    Args:
        tensor: torch.Tensor of shape (B, C, H, W, D)
        crop_size: tuple of 3 ints (h, w, d)
    
    Returns:
        Cropped tensor of shape (B, C, crop_h, crop_w, crop_d)
    """
    _, h, w, d = tensor.shape
    ch, cw, cd = crop_size

    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    start_d = (d - cd) // 2


    return tensor[:, start_h:start_h+ch, start_w:start_w+cw, start_d:start_d+cd]


def select_files(folder_path, prefix, suffix):
    selected = [
        f for f in os.listdir(folder_path)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    return sorted(selected)



class PairDataset(Dataset):
    def __init__(self, root_dir,cases_all,meta_data_dir,subjects_data_file,crop_size transform=None):
        self.root_dir = root_dir
        self.cases_all = cases_all
        self.transform = transform
        self.meta_data_dir = meta_data_dir
        self.meta_file = subjects_data_file
        self.crop_size = crop_size
        self.pairs = self._find_visit_pairs()
        self.data_len = len(self.pairs)
    
    def _get_sex_info(self,par_i):
         
        tsv_path = os.path.join(self.meta_file)
        tsv_df = pd.read_csv(tsv_path, sep='\t')
        sex = tsv_df.loc[tsv_df['par'] == par_i, 'sex']
        return sex.item() 

    def _get_meta_data(self, tsv_df, sex, visit1_name, visit2_name,max_delta,max_age):

        # Age at baseline (visit1)
        age1 = tsv_df.loc[
            tsv_df['source_session'] == visit1_name, 'age'
        ].values[0]
        # MMSE at baseline
        mmse = tsv_df.loc[
            tsv_df['source_session'] == visit1_name, 'mmse'
        ].values[0]
        # Diagnosis at baseline
        diagnosis = tsv_df.loc[
            tsv_df['source_session'] == visit1_name, 'dx1'
        ].values[0]

        # Age at follow-up (visit2) → for delta_time
        age2 = tsv_df.loc[
            tsv_df['source_session] == visit2_name, 'age'
        ].values[0]

        # Delta time (years between visits)
        delta_time = float(age2) - float(age1)
            
        # Max age in dataset
        #max_age = tsv_df['age'].max()

        # Max delta_time across dataset (max age difference per subject/session pairs)
        #max_delta_time = tsv_df['age'].max() - tsv_df['age'].min()

        return {
            'age': float(age1),
            'mmse': float(mmse),
            'diagnosis': str(diagnosis),
            'sex': str(sex),
            'delta_time': delta_time,
            'max_age': float(max_age),
            'max_delta_time': float(max_delta),
            
        }

        
    def _get_age_tsv(self,case_dir,visit_1_dir,visit_2_dir):
        meta_data_path = os.path.join(self.meta_data_dir,case_dir)
        tsv_df, _ = self.load_first_tsv_from_dir(meta_data_path)
        visit1_name =  visit_1_dir.split('-')[-1]
        visit2_name =  visit_2_dir.split('-')[-1]
        age1_row = tsv_df.loc[tsv_df['source_session'] == visit1_name, 'age']

        if not age1_row.empty:
            age1 = age1_row.values[0]
        else:
            age1 = None
        age2_row = tsv_df.loc[tsv_df['source_session'] == visit2_name, 'age']
        if not age2_row.empty:
            age2 = age2_row.values[0]
        else:
            age2 = None
        return age1,age2
        
    def _find_visit_pairs(self):
        pairs = []
        all_deltas = []
        all_ages = []
        max_delta = 0
        max_age = 0
        for case_name in self.cases_all: #os.listdir(self.root_dir)
            case_path = os.path.join(self.root_dir, case_name)
            if not os.path.isdir(case_path) or not os.path.isdir(os.path.join(self.meta_data_dir,case_name)):
                continue
            visit_names = sorted([
                v for v in os.listdir(case_path)
                if os.path.isdir(os.path.join(case_path, v))
            ])
            if len(visit_names) < 2:
                continue  # Skip cases with < 2 visits
            # Generate all unique visit pairs
            for visit1, visit2 in itertools.combinations(visit_names, 2):
                age1,age2 = self._get_age_tsv(case_name,visit1,visit2)
                if age1 is None or age2 is None:
                    continue
                if np.isnan(age1) or np.isnan(age2):
                    continue
                delta_age = age2 - age1
                max_delta = max(delta_age,max_delta)
                max_age = max(age1,age2,max_age)
                all_deltas.append(delta_age)
                all_ages.extend([age1, age2])
                # Skip if delta is too small
                if delta_age < 1: #6,4 (used in last time)
                    continue
                pairs.append({
                    'case': case_name,
                    'visit1': os.path.join(case_path, visit1),
                    'visit2': os.path.join(case_path, visit2)

                })
  
        max_delta = max(all_deltas) if all_deltas else 0
        max_age = max(all_ages) if all_ages else 0

        # attach global max to each pair
        for p in pairs:
            p['max_delta_t'] = max_delta
            p['max_age'] = max_age
        return pairs
    
    def load_first_tsv_from_dir(self,root_dir):
        # Find all files ending with .tsv
        tsv_files = [
            f for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f)) and f.endswith('.tsv')
        ]
        
        if not tsv_files:
            raise FileNotFoundError(f"No TSV file found in {root_dir}")
        
        # Pick the first one (or add logic if you want to choose differently)
        tsv_path = os.path.join(root_dir, tsv_files[0])
        
        # Load the TSV
        df = pd.read_csv(tsv_path, sep='\t')
        
        return df, tsv_path

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_info = self.pairs[idx]
        max_delta_t = pair_info['max_delta_t']
        # Here, replace with your own image loading code:
        img1,seg1 = self._load_image(pair_info['visit1'])
        img2,seg2 = self._load_image(pair_info['visit2'])
        #if seg1 is None or seg2 is None:
            #return self.__getitem__((idx + 1) % len(self.pairs))
        case_dir = pair_info['case']
        meta_data_path = os.path.join(self.meta_data_dir,case_dir)
        tsv_df, _ = self.load_first_tsv_from_dir(meta_data_path)
        visit1_name =  os.path.basename(os.path.normpath(pair_info['visit1'])).split('-')[-1]
        visit2_name =  os.path.basename(os.path.normpath(pair_info['visit2'])).split('-')[-1]
        age_visit1 = tsv_df.loc[tsv_df['source_session'] == visit1_name, 'age'].values[0]
        age_visit2 = tsv_df.loc[tsv_df['source_session'] == visit2_name, 'age'].values[0]
        delta_t = age_visit2 - age_visit1
        sex = self._get_sex_info(case_dir)
        meta_data_dict = self._get_meta_data(tsv_df, sex, visit1_name, visit2_name,max_delta_t,pair_info['max_age'])
        meta_data_dict["id"] = case_dir


 
        img1 =  center_crop_3d(img1, self.crop_size)
        img2=  center_crop_3d(img2, self.crop_size)
        seg1 =  center_crop_3d(seg1, self.crop_size)
        seg2 =  center_crop_3d(seg2, self.crop_size)


        img1 = norm_images(img1)
        img2 = norm_images(img2)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(delta_t).unsqueeze(0), torch.tensor(max_delta_t).unsqueeze(0),seg1,seg2,meta_data_dict #dtype=torch.float64


    def _load_image(self, visit_path):
 
        image_files = os.listdir(os.path.join(visit_path))
        image_file_selected = [
        f for f in image_files
        if  "mni_norm" in f 
        ]
        if len(image_file_selected)>1:
            image_file_selected = [
                    f for f in image_file_selected
                    if "run-01_" in f 
                    ]
        mask_file_selected = [
        f for f in image_files
        if  "seg" in f 
        ]
        if len(mask_file_selected)>1:
            mask_file_selected = [
                    f for f in mask_file_selected
                    if "run-01_" in f or ( "_masked_seg" not in f)
                    ]
        image = nib.load(os.path.join(visit_path,image_file_selected[0])).get_fdata() 
        mask_exists = len(mask_file_selected) > 0
        seg = None
        if mask_exists:
            seg = nib.load(os.path.join(visit_path,mask_file_selected[0])).get_fdata() 

        return torch.tensor(image).unsqueeze(0).float(),torch.tensor(seg).unsqueeze(0).float() 
      


