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
from monai.transforms import Resize
class CapDataset(Dataset):
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
        set_track_meta(False)

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
        if nifti_data.ndim == 4 :
            nifti_data = nifti_data[..., 0]
        elif nifti_data.ndim != 3 :
            raise ValueError(f"Unexpected NIfTI data shape {nifti_data.shape} from {nifti_path}")
        return nifti_data
        # .transpose(2, 0, 1)[np.newaxis, ...]

    def split_image(self, image):
        # Split the image into smaller parts (c x 32 x 256 x 256)
        parts = []
        for i in range(0, image.shape[0], 32):
            part = image[i:i+32, :, :]
            if part.shape[0] == 32:
                parts.append(part)
        return parts
    def resize_volume(self,volume,slices=None):
        volume = volume[np.newaxis, ...]
        if slices is not None:
            target_shape = [slices, 256, 256]
        else:
            target_shape = [volume.shape[1], 256, 256]
        resize_transform = Resize(spatial_size=target_shape, mode="bilinear")
        resized_volume = resize_transform(volume)
        resized_volume = resized_volume.squeeze(0).numpy()
        return resized_volume
    def slice_volume(self,volume,mean_slices=256):
        # Calculate the number of slices to pad if needed
        if volume.shape[0] < mean_slices:
            pad_slices = mean_slices - volume.shape[0]
            pad_before = pad_slices // 2
            pad_after = pad_slices - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode ='edge') # mode='constant', constant_values=0)
            return volume
        # Get the center of the volume
        center = volume.shape[0] // 2
        # Get the range of slices
        start = center - mean_slices // 2
        end = center + mean_slices // 2
        # Get the volume with the mean slices
        volume = volume[start:end, ...]
        return volume
    def converttoaxial(self,nifti_arr,rotations=-1,t=1):
        axial_volume = list()
        for i in range(nifti_arr.shape[2]):
            if t == 1:
                axial_slice = np.rot90(nifti_arr[:,:,i].T, k=rotations)
            else:
                axial_slice = np.rot90(nifti_arr[:,:,i], k=rotations)
            axial_volume.append(np.expand_dims(axial_slice, axis=0))
        axial_volume = np.concatenate(axial_volume)
        return axial_volume
    def normalise(self, image):
        np_img = image
        np_img = np.clip(np_img, -150., 250.).astype(np.float32) 
        return np_img

    def preprocess_image(self, image):
        transform = mtf.Compose([
            # mtf.Spacing(pixdim=[1,1,1],mode='bilinear'),
            mtf.ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
            mtf.CropForeground(allow_smaller=True),
            mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
        ])
        image = transform(image)
        return image.numpy()
    def process(self, image_path, orientation,mean_slices=256):
        image = self.get_image_from_nifti(image_path)
        if orientation == 'Coronal':
            image_ax = self.converttoaxial(image, rotations=-1, t=1)
        elif orientation == 'Sagittal':
            image_ax = self.converttoaxial(image,rotations=-1,t=1)
        else:
            image = image.transpose(2, 0, 1)
            image_ax = image
        total_slices = image_ax.shape[0]
        # if total_slices > mean_slices:
        #     image_ax = self.slice_volume(image_ax,mean_slices=mean_slices)
        image_ax = self.slice_volume(image_ax,mean_slices=mean_slices)
        # image_ax = self.resize_volume(image_ax)
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
                image_parts = [self.preprocess_image(part[np.newaxis, ...]) for part in image_parts]
                image_parts = [self.transform(part) for part in image_parts]

                # Convert image_parts to a tensor and add a new axis for parts dimension
                image_parts = torch.stack(image_parts, dim=0)

                answer = data["Impression"]

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
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