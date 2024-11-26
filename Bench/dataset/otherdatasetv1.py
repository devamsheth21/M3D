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
        return nifti_data.transpose(2, 0, 1)[np.newaxis, ...]
    def normalise(self, image):
        np_img = image
        np_img = np.clip(np_img, -150., 250.).astype(np.float32) 
        return np_img

    def preprocess_image(self, image):
        transform = mtf.Compose([
            mtf.ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
            mtf.CropForeground(allow_smaller=True),
            mtf.Resize(spatial_size=[32, 256, 256], mode="bilinear")
        ])
        image = self.normalise(image)
        image = transform(image)
        return image.numpy()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.df.iloc[idx]
                image_path = data["Image"]

                image = self.get_image_from_nifti(image_path)
                image = self.preprocess_image(image)
                image = self.transform(image)
                
                # image = None
                # image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized

                answer = data["Text"]

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
                    'image': image,
                    'input_id': input_id,
                    'prompt_question': prompt_question,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption-0",
                    'acc': data['acc'],
                    'label': label
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)
    
    def is_valid_nifti(self, nifti_path):
        try:
            nifti_data = nib.load(nifti_path).get_fdata()
            if nifti_data.ndim == 4:
                return True  # You can modify this if you want to handle 4D specifically
            elif nifti_data.ndim == 3:
                return True
            else:
                return False
        except (nib.filebasedimages.ImageFileError, ValueError):
            if self.logger:
                self.logger.warning(f"Invalid NIfTI file: {nifti_path}")
            return False
        except np.exceptions.DTypePromotionError:
            if self.logger:
                self.logger.warning(f"Invalid NIfTI data type: {nifti_path}")
            return False
        except FileNotFoundError:
            if self.logger:
                self.logger.warning(f"File not found: {nifti_path}")
            return False
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Unknown error: {e}")
            return False
