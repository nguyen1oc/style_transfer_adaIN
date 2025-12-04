import os
import glob
import numpy as np
from tqdm import tqdm
from skimage import io, transform 
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


trans = transforms.Compose([transforms.RandomCrop(256),
                            transforms.ToTensor(),
                            normalize])

def denorm(tensor, device='cpu'):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1).to(device)
    return torch.clamp(tensor * std + mean, 0, 1)

class StyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_dir, crop_size=256, resized_size=512, transform=trans):
        self.crop_size = crop_size
        self.resized_size = resized_size
        self.transform = transform


        self.content_resized = os.path.join('/kaggle/working', 'train2017_resized')
        self.style_resized   = os.path.join('/kaggle/working', 'images_resized')


        if not os.path.exists(self.content_resized):
            os.makedirs(self.content_resized, exist_ok=True)
            self._resize_folder(content_dir, self.content_resized)
        if not os.path.exists(self.style_resized):
            os.makedirs(self.style_resized, exist_ok=True)
            self._resize_folder(style_dir, self.style_resized, subfolders=True)

        self.content_images = glob.glob(os.path.join(self.content_resized, '*'))
        self.style_images = glob.glob(os.path.join(self.style_resized, '*', '*'))

        np.random.shuffle(self.content_images)
        np.random.shuffle(self.style_images)


        self.image_pairs = list(zip(self.content_images, self.style_images))

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                normalize
            ])

    def _resize_folder(self, src_dir, dst_dir, subfolders=False):
        print(f"Resizing images from {src_dir} -> {dst_dir}")
        if subfolders:
            for sub in os.listdir(src_dir):
                src_sub = os.path.join(src_dir, sub)
                dst_sub = os.path.join(dst_dir, sub)
                os.makedirs(dst_sub, exist_ok=True)
                for img_name in tqdm(os.listdir(src_sub)):
                    self._resize_image(os.path.join(src_sub, img_name), os.path.join(dst_sub, img_name))
        else:
            for img_name in tqdm(os.listdir(src_dir)):
                self._resize_image(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))

    def _resize_image(self, src_path, dst_path):
        try:
            img = io.imread(src_path)
            if len(img.shape) != 3 or img.shape[2] != 3:
                return  # skip grayscale
            H, W, _ = img.shape
            if H < W:
                ratio = W / H
                H_new = self.resized_size
                W_new = int(ratio * H_new)
            else:
                ratio = H / W
                W_new = self.resized_size
                H_new = int(ratio * W_new)
            img_resized = transform.resize(img, (H_new, W_new), mode='reflect', anti_aliasing=True)
            io.imsave(dst_path, (img_resized * 255).astype(np.uint8))
        except:
            pass

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        content_path, style_path = self.image_pairs[idx]
        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')

        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)

        return content_img, style_img

