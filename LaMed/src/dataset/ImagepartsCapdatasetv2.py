import os
import numpy as np
from PIL import Image
import monai.transforms as mtf
import pandas as pd
import random
import torch
import nibabel as nib
from torch.utils.data import Dataset
from monai.data import set_track_meta
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from .prompt_templates import Caption_templates

class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="test", logger=None):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.logger = logger

        self.image_tokens = "<im_patch>" * args.proj_out_num
        if args.cap_data_path:
            self.df = pd.read_csv(args.cap_data_path)
            self.df.dropna(subset=['Image','Findings'], inplace=True)
        else:
            self.df = args.df
        self.df = self.df[self.df['mode'] == mode]
        if True:
            self.df = self.df.sample(400, random_state=42).reset_index(drop=True)
        if logger:
            logger.info(f"Data size: {len(self.df)}")
        if 'Findings' in self.df.columns:
            self.df.rename(columns={'Findings': 'Text'}, inplace=True)
        if 'AccessionNumber' in self.df.columns:
            self.df.rename(columns={'AccessionNumber': 'acc'}, inplace=True)
        self.df.dropna(subset=['Text'], inplace=True)
        if self.df.duplicated().sum() > 0:
            print(f"Duplicate data: {self.df.duplicated().sum()}")
            self.df.drop_duplicates(keep='first', inplace=True)
        self.caption_prompts = Caption_templates
        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        # set_track_meta(False)
        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform
        self.preprocess = mtf.Compose(
            [
                mtf.LoadImage(image_only=True),
                mtf.EnsureChannelFirst(channel_dim="no_channel"),
                mtf.Orientation(axcodes="RAS"),
                # lambda x: (print(f"Shape before spacing: {x.shape}"), x)[1],  # Print shape before spacing
                mtf.Spacing(pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                # lambda x: (print(f"Shape after spacing: {x.shape}"), x)[1],  # Print shape after spacing
                mtf.ScaleIntensityRange(a_min=-150, a_max=250, b_min=0, b_max=1.0, clip=True),
                mtf.NormalizeIntensity(),
                mtf.CropForeground(),
                mtf.Resize((256, 256, None), mode='bilinear'),
                # lambda x: (print(f"Shape after Resize: {x.shape}"), x)[1],  # Print shape before spacing
                mtf.ToTensor(dtype=torch.float),

            ])
        # def print_shape(x , after =None):
        #     if isinstance(x, dict):
        #         print(f"Shape : {x["Image"]} after {after}")
        #     print(f"Shape: {x.shape} after {after}")
        #     return x
        # def depthfirst(image_tensor, orientation):
        #     if orientation.lower() == 'axial':
        #         image_tensor = image_tensor.permute(0, 3, 1, 2)
        #     elif orientation.lower() == 'coronal':
        #         image_tensor = image_tensor.permute(0, 2, 1, 3)
        #     elif orientation.lower() == 'sagittal':
        #         # image_tensor = image_tensor.permute(0, 1, 3, 2)
        #     return image_tensor
        # self.preprocess2 = mtf.Compose(
        #     [
        #         mtf.LoadImaged(keys=["Image"],image_only=False ,reader = NibabelReader),
        #         print_shape,
        #         mtf.EnsureChannelFirstd(keys=["Image"],channel_dim="no_channel"),
        #         mtf.Orientationd(keys=["Image"], axcodes="RAS"),
        #         mtf.Spacingd(keys=["Image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        #         print_shape,
        #         mtf.ScaleIntensityRanged(keys=["Image"], a_min=-150, a_max=250, b_min=0, b_max=1.0, clip=True),
        #         mtf.NormalizeIntensityd(keys=["Image"]),
        #         mtf.CropForegroundd(keys=["Image"], source_key="Image"),
        #         print_shape,
        #         mtf.Lambda(lambda x: depthfirst(x['Image'], x['Orientation'])),
        #         mtf.Resize(spatial_size=(None, 256, 256), mode='bilinear'),
        #         print_shape,
        #     ])

    def convert_to_axial_torch(self,image_tensor, rotations=-1, t=1 , orientation = None):
        if orientation.lower() == 'axial':
            axial_volume = image_tensor.squeeze(0).permute(2,0,1)
        else:
            axial_volume = []
            for i in range(image_tensor.shape[3]):
                if t == 1:
                    axial_slice = torch.rot90(image_tensor[0, :, :, i].T, k=rotations, dims=(0, 1))
                else:
                    axial_slice = torch.rot90(image_tensor[0, :, :, i], k=rotations, dims=(0, 1))
                axial_volume.append(axial_slice.unsqueeze(0))
            axial_volume = torch.cat(axial_volume)
        return axial_volume

    def split_into_equal_parts(self,image_tensor, part_size=32, num_parts=8):
        """
        Split tensor into parts with efficient padding and centering.
        
        Args:
            image_tensor (torch.Tensor): Input tensor of shape [D, H, W]
            part_size (int, optional): Desired size of each part. Defaults to 32.
            num_parts (int, optional): Total number of parts to create. Defaults to 8.
        
        Returns:
            torch.Tensor: Stacked tensor of shape [num_parts, 1, part_size, H, W]
        """
        # Total depth of the input tensor
        total_depth = image_tensor.shape[0]
        
        # Center the split if the tensor is larger than num_parts * part_size
        if total_depth > num_parts * part_size:
            start_idx = (total_depth - num_parts * part_size) // 2
            image_tensor = image_tensor[start_idx:start_idx + num_parts * part_size]
            total_depth = image_tensor.shape[0]
        
            # Split the tensor
            split_tensor = list(torch.split(image_tensor, part_size, dim=0))
        else:
            split_tensor = list(torch.split(image_tensor, part_size, dim=0) )    
            # Check if the last part needs padding
            if len(split_tensor) > 0 and split_tensor[-1].shape[0] < part_size:
                # Get the last part that is incomplete
                last_part = split_tensor[-1]
                
                # Create padding by repeating the last frame
                padding_size = part_size - last_part.shape[0]
                edge_padding = last_part[-1].repeat(padding_size, *[1] * (last_part.ndim - 1))
                
                # Concatenate the original part with edge padding
                padded_last_part = torch.cat([last_part, edge_padding], dim=0)
                
                # Replace the last part in the split_tensor
                split_tensor = list(split_tensor)[:-1] + [padded_last_part]
            
            # If not enough parts, pad with zeros
            while len(split_tensor) < num_parts:
                zero_padding = torch.zeros(
                    (part_size, *split_tensor[0].shape[1:]), 
                    dtype=image_tensor.dtype, 
                    device=image_tensor.device
                )
                split_tensor.append(zero_padding)
        # Stack the parts and add channel dimension
        stacked_tensor = torch.stack(split_tensor[:num_parts], dim=0).unsqueeze(1)
    
        return stacked_tensor
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_attempts = 1
        for _ in range(max_attempts):
            try:
                data = self.df.iloc[idx]
                image_path = data["Image"]
                orientation = data["Orientation"]
                image = self.preprocess(image_path)
                image = image.as_tensor()
                # print(image.shape,"after preprocess")
                image_ax = self.convert_to_axial_torch(image, orientation=orientation)
                # print(image_ax.shape,"after convert to axial")
                image_parts = self.split_into_equal_parts(image_ax, part_size=32, num_parts=8)
                answer = data["Text"]

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + ' ' + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image_parts,
                    'input_id': input_id,
                    'prompt_question': prompt_question,
                    'attention_mask': attention_mask,
                    'question': question_tensor,
                    'answer': answer,
                    'question_type': "Caption-0",
                    'image_path': data['Image'],
                    'acc': data['acc'],
                    'label': label
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image_parts)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.df) - 1)

class CapDataset_prev(Dataset):
    def __init__(self, args, tokenizer, mode="test", logger=None):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.logger = logger

        self.image_tokens = "<im_patch>" * args.proj_out_num
        if args.cap_data_path:
            self.df = pd.read_csv(args.cap_data_path)
        else:
            self.df = args.df
        self.df = self.df[self.df['mode'] == mode]
        if True:
            self.df = self.df.sample(400, random_state=42).reset_index(drop=True)
        if logger:
            logger.info(f"Data size: {len(self.df)}")
        if 'Findings' in self.df.columns:
            self.df.rename(columns={'Findings': 'Text'}, inplace=True)
        if 'AccessionNumber' in self.df.columns:
            self.df.rename(columns={'AccessionNumber': 'acc'}, inplace=True)
        self.df.dropna(subset=['Text'], inplace=True)
        if self.df.duplicated().sum() > 0:
            print(f"Duplicate data: {self.df.duplicated().sum()}")
            self.df.drop_duplicates(keep='first', inplace=True)

        self.caption_prompts = Caption_templates

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        # set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def get_image_from_nifti(self, nifti_path):
        nifti_data = nib.load(nifti_path).get_fdata()
        if nifti_data is None:
            raise ValueError(f"Failed to load NIfTI data from {nifti_path}")
        if nifti_data.ndim == 4:
            nifti_data = nifti_data[..., 0]
        elif nifti_data.ndim != 3:
            raise ValueError(f"Unexpected NIfTI data shape {nifti_data.shape} from {nifti_path}")
        return nifti_data

    def split_image(self, image):
        # Split the image into smaller parts (can be less than 32 slices)
        parts = []
        for i in range(0, image.shape[0], 32):
            part = image[i:i+32, :, :]
            parts.append(part)
        return parts

    def resize_volume(self, volume, slices=None):
        volume = volume[np.newaxis, ...]
        if slices is not None:
            target_shape = [slices, 256, 256]
        else:
            target_shape = [volume.shape[1], 256, 256]
        resize_transform = Resize(spatial_size=target_shape, mode="bilinear")
        resized_volume = resize_transform(volume)
        resized_volume = resized_volume.squeeze(0).numpy()
        return resized_volume

    def slice_volume(self, volume, mean_slices=256):
        # Calculate the number of slices to pad if needed
        if volume.shape[0] < mean_slices:
            pad_slices = mean_slices - volume.shape[0]
            pad_before = pad_slices // 2
            pad_after = pad_slices - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')
            return volume
        # Get the center of the volume
        center = volume.shape[0] // 2
        # Get the range of slices
        start = center - mean_slices // 2
        end = center + mean_slices // 2
        # Get the volume with the mean slices
        volume = volume[start:end, ...]
        return volume

    def converttoaxial(self, nifti_arr, rotations=-1, t=1):
        axial_volume = []
        for i in range(nifti_arr.shape[2]):
            if t == 1:
                axial_slice = np.rot90(nifti_arr[:, :, i].T, k=rotations)
            else:
                axial_slice = np.rot90(nifti_arr[:, :, i], k=rotations)
            axial_volume.append(np.expand_dims(axial_slice, axis=0))
        axial_volume = np.concatenate(axial_volume)
        return axial_volume

    def normalise(self, image):
        np_img = image
        np_img = np.clip(np_img, -150., 250.).astype(np.float32)
        return np_img

    def preprocess_image(self, image):
        transform = mtf.Compose([
            mtf.ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
            mtf.CropForeground(allow_smaller=True),
            mtf.Resize(spatial_size=[None, 256, 256], mode="bilinear")
        ])
        image = transform(image)
        return image.numpy()

    def process(self, image_path, orientation, mean_slices=256):
        image = self.get_image_from_nifti(image_path)
        if orientation == 'Coronal':
            image_ax = self.converttoaxial(image, rotations=-1, t=1)
        elif orientation == 'Sagittal':
            image_ax = self.converttoaxial(image, rotations=-1, t=1)
        else:
            image = image.transpose(2, 0, 1)
            image_ax = image
        total_slices = image_ax.shape[0]
        if total_slices > mean_slices:
            image_ax = self.slice_volume(image_ax, mean_slices=mean_slices)
        # Do not pad here; we'll handle padding after preprocessing
        image_ax = self.normalise(image_ax)
        image_parts = self.split_image(image_ax)
        return image_parts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_attempts = 1
        for _ in range(max_attempts):
            try:
                data = self.df.iloc[idx]
                image_path = data["Image"]

                image_parts = self.process(image_path, data['Orientation'], mean_slices=256)
                # Preprocess image parts
            
                image_parts = [self.preprocess_image(part[np.newaxis, ...]) for part in image_parts]
                # Add padding to incomplete and zero parts after preprocessing
                padded_image_parts = []
                for part in image_parts:
                    # Check for incomplete parts
                    part = part.squeeze(0)
                    if part.shape[0] < 32:
                        pad_slices = 32 - part.shape[0]
                        pad_before = pad_slices // 2
                        pad_after = pad_slices - pad_before
                        part = np.pad(
                            part,
                            pad_width=((pad_before, pad_after), (0, 0), (0, 0)),
                            mode='edge'
                        )
                    # Check for zero parts (all zeros)
                    if not np.any(part):
                        # Replace zero part with small constant to avoid issues
                        part += 1e-5
                    padded_image_parts.append(part)
                image_parts = [mtf.ToTensor(dtype=torch.float)(part) for part in padded_image_parts]
                image_parts = torch.stack(image_parts, dim=0)
                image_parts = self.transform(image_parts)

                if image_parts.shape[0] < 8:
                    pad_part = torch.zeros_like(image_parts[0])
                    image_parts = torch.cat([image_parts, pad_part.repeat(8 - image_parts.shape[0], 1, 1, 1)], dim=0)


                answer = data["Impression"]

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + ' ' + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image_parts,
                    'input_id': input_id,
                    'prompt_question': prompt_question,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption-0",
                    'image_path': data['Image'],
                    'acc': data['acc'],
                    'label': label
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image_parts)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.df) - 1)