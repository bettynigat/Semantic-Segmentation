import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from PIL import Image
import urllib
import glob
import skimage.io as skio
from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class Relabel:
    def __init__(self, inlabel, outlabel):
        self.inlabel = inlabel
        self.outlabel = outlabel

    def __call__(self, tensor):
        tensor[tensor == self.inlabel] = self.outlabel
        return tensor

class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(2):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

def dataAugment(root):
    images_root = os.path.join(root, 'Images')
    labels_root = os.path.join(root, 'Labels')

    filenames = [image_basename(f)
        for f in os.listdir(labels_root) if is_image(f)]

    filenames.sort()
    angle_1 = 180
    angle_2 = 90
    angle_3 = 270
    #i = 0
    for filename in filenames:
        im = Image.open(images_root + '/' + filename + '.jpg')
        # if (i%2==0):
        out = im.rotate (angle_1)
        out_2 = im.rotate(angle_3)
        #  else:
        out_3 = im.rotate (angle_2)
        out_4 = im.transpose(Image.FLIP_LEFT_RIGHT)
        out.save(images_root + '/12_' + filename + '.jpg')
        out_2.save(images_root + '/2_' + filename + '.jpg')
        out_3.save(images_root + '/5_' + filename + '.jpg')
        out_4.save(images_root + '/32_' + filename + '.jpg')
        
        im = Image.open(labels_root + '/' + filename + '.png')
        #if (i%2==0):
        out = im.rotate (angle_1)
        out_2 = im.rotate(angle_3)
        #else:
        out_3 = im.rotate (angle_2)
        out_4 = im.transpose(Image.FLIP_LEFT_RIGHT)
        out.save(labels_root + '/12_' + filename + '.png')
        out_2.save(labels_root + '/2_' + filename + '.png')
        out_3.save(labels_root + '/5_' + filename + '.png')
        out_4.save(labels_root + '/32_' + filename + '.png')
        #i+=1
    return

class Colorize:
    def __init__(self, n=9):
        self.cmap = colormap(10)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])


    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

class CoE_Test_Dataset(Dataset):

   def __init__(self, root, input_transform=None):
      self.images_root = os.path.join(root, 'Images')

      self.filenames = [image_basename(f)
         for f in os.listdir(self.images_root) if is_image(f)]

      self.filenames.sort()

      self.input_transform = input_transform

   def __getitem__(self, index):
      filename = self.filenames[index]
      with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
         image = load_image(f).convert('RGB')

      if self.input_transform is not None:
         image = self.input_transform(image)

      return image, filename

   def __len__(self):
      return len(self.filenames)
class CoE_Dataset(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, train=False):
        self.images_root = os.path.join(root, 'Images')
        self.labels_root = os.path.join(root, 'Labels')
        
        self.train=train

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]

        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.train:
            crop_size = 128
            w, h = image.size
            if (w < 128):
                x1 = 0
            else: 
                x1 = random.randint(0, w - crop_size)
            if(h < 128):
                y1 = 0
            else:
                y1 = random.randint(0, h - crop_size)
    
            image = image.crop((x1, y1, x1+crop_size, y1+crop_size))
            label = label.crop((x1, y1, x1+crop_size, y1+crop_size))
        
        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, filename

    def __len__(self):
        return len(self.filenames)


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    epsilon = 1e-12
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    
    return area_inter.float(), area_union.float() + epsilon

