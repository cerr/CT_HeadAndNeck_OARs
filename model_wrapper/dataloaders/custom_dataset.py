# Author: Aditi Iyer
# Email: iyera@mskcc.org
# Date: Feb 10, 2020
# Description: Custom dataloader for chewing and swallowing structures.

import os
import numpy as np
import scipy.misc as m
from PIL import Image
import h5py
from torch.utils import data
from torchvision import transforms
from skimage.transform import resize
from dataloaders import custom_transforms as tr
import SimpleITK as sitk

class struct(data.Dataset):

    def __init__(self, args):

        self.root = args.dataPath

        self.args = args
        self.files = {}

        self.files = self.glob(rootdir=self.root, suffix='.nii.gz')
        scanImg = sitk.ReadImage(self.files[0])
        scanArr = np.flip(np.transpose(sitk.GetArrayFromImage(scanImg),(1,2,0)),axis=2)
        self.scan = scanArr
        print("Found %d images" % (self.__len__()))

    def __len__(self):
        #return len(self.files)
        return self.scan.shape[2]

    def __getitem__(self, index):

        _img, _imagesize = self._load_image(index)
        sample = {'image': _img, 'imagesize': _imagesize, 'fname': self.files[0]}
    
        #Mean & std dev. normalization
        return self.transform_ts(sample)

    def _load_image(self, index):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        inputSize = 320
        scanArr = self.scan

        if index==0:
            image = np.stack((scanArr[:,:,index],
                 scanArr[:, :, index],
                 scanArr[:, :, index+1]),axis=2)

        elif  index==self.__len__()-1:
            image = np.stack((scanArr[:, :, index-1],
                     scanArr[:, :, index],
                     scanArr[:, :, index]),axis=2)
        else:
            image = np.stack((scanArr[:, :, index-1],
                     scanArr[:, :, index],
                     scanArr[:, :, index+1]),axis=2)

        imagesize = (scanArr.shape[0],scanArr.shape[1])

        #Resize image
        image = resize(image, (inputSize,inputSize), anti_aliasing = True)

        #Normalize image from 0-255 (to match pre-trained dataset of RGB images with intensity range 0-255)
        image = (255*(image - np.min(image)) / np.ptp(image).astype(int)).astype(np.uint8)
        image = Image.fromarray(image.astype(np.uint8))

        return image, imagesize

    def glob(self, rootdir='.', suffix=''):
        """Performs glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(rootdir, filename)
            for filename in os.listdir(rootdir) if filename.endswith(suffix)]

    def transform_ts(self, sample):
        
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.cropSize),
            tr.Normalize(mean=self.args.mean, std=self.args.std),
            tr.ToTensor()])

        return composed_transforms(sample)

