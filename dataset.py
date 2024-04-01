import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os.path
import torch.utils.data as data

import sys
sys.path.append('/home/weichen/projects/visobj/related-projects/PerceptualSimilarity')
all_data_path='/home/weichen/projects/visobj/proposals/mise/data/ulgn/all_data.pt'

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        
    def name(self):
        return 'BaseDataset'
    
    def initialize(self):
        pass

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NP_EXTENSIONS = ['.npy',]

def is_image_file(filename, mode='img'):
    if(mode=='img'):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif(mode=='np'):
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)

def make_dataset(dirs, mode='img'):
    if(not isinstance(dirs,list)):
        dirs = [dirs,]

    images = []
    for dir in dirs:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    images.append(path)

    # print("Found %i images in %s"%(len(images),root))
    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def get_preprocess_fn(preprocess, load_size, interpolation):
    if preprocess == "LPIPS":
        t = transforms.ToTensor()
        return lambda pil_img: t(pil_img.convert("RGB")) / 0.5 - 1.
    else:
        if preprocess == "DEFAULT":
            t = transforms.Compose([
                transforms.Resize((load_size, load_size), interpolation=interpolation),
                transforms.ToTensor()
            ])
        elif preprocess == "DISTS":
            t = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
        elif preprocess == "SSIM" or preprocess == "PSNR":
            t = transforms.ToTensor()
        else:
            raise ValueError("Unknown preprocessing method")
        return lambda pil_img: t(pil_img.convert("RGB"))
    

# image
class ImageData(Dataset):
    """
    path: Path to the directory where the images are saved. 
        Images are assumed to be saved in the format <label_name>/image.jpg 
        (e.g., zucchini/zucchini_08n.jpg)
    transform: A torchvision.transforms object that will be applied to the images
    """

    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.images = []
        self.labels = []
        self.categories = []

        # Look through each sub-directory in the path
        for label in os.listdir(self.path):
            for image in os.listdir(os.path.join(self.path, label)):
                self.images.append(os.path.join(self.path, label, image))
                self.labels.append(label)
            self.categories.append(label)

        self.categories = sorted(self.categories)
        self.label_to_index = {self.categories[i]: i for i in range(len(self.categories))}

    def __len__(self):
        # Return the total number of images
        return len(self.images)

    def __getitem__(self, idx):
        # Open and send one image and its label
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
           image = transforms.ToTensor()(image)
            
        label = self.label_to_index[self.labels[idx]]
        return image, label
    
def read_properties(tsv_path):
    # Read the TSV file into a DataFrame with 'uniqueID' as the index
    properties_df = pd.read_csv(tsv_path, sep='\t', index_col='uniqueID')
    # Convert the DataFrame into a dictionary with 'uniqueID' as keys
    properties_dict = properties_df.to_dict(orient='index')
    return properties_dict


# image
class ImageDataPlus(Dataset):
    """
    path: Path to the directory where the images are saved. 
    transform: A torchvision.transforms object that will be applied to the images
    """

    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.images = []
        self.labels = []
        self.categories = []

        # Look through each sub-directory in the path
        for images in os.listdir(self.path):
            if images[-4:] == '.jpg':
                self.labels.append(images[:-4])
                self.images.append(os.path.join(self.path, images))
                self.categories.append(images[:-4])

        self.categories = sorted(self.categories)
        self.label_to_index = {self.categories[i]: i for i in range(len(self.categories))}

    def __len__(self):
        # Return the total number of images
        return len(self.images)

    def __getitem__(self, idx):
        # Open and send one image and its label
        image_path = self.images[idx]
        image =  Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        label = self.label_to_index[self.labels[idx]]
        return image, label