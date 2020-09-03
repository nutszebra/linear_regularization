from PIL import Image
import os
import os.path
import numpy as np
import pickle

from torchvision.datasets import VisionDataset


class CIFAR10C(VisionDataset):

    # files = ['brightness.npy', 'contrast.npy', 'defocus_blur.npy',
    #          'elastic_transform.npy', 'fog.npy', 'frost.npy',
    #          'gaussian_blur.npy', 'gaussian_noise.npy', 'glass_blur.npy',
    #          'impulse_noise.npy', 'jpeg_compression.npy', 'motion_blur.npy',
    #          'pixelate.npy', 'saturate.npy', 'shot_noise.npy',
    #          'snow.npy', 'spatter.npy', 'speckle_noise.npy',
    #          'zoom_blur.npy'] 
    files = ['gaussian_noise.npy', 'shot_noise.npy', 'impulse_noise.npy', 'defocus_blur.npy',
             'glass_blur.npy', 'motion_blur.npy', 'zoom_blur.npy', 'snow.npy', 'frost.npy', 'fog.npy',
             'brightness.npy', 'contrast.npy', 'elastic_transform.npy', 'pixelate.npy',
             'jpeg_compression.npy'
             ]

    labels = 'labels.npy'

    

    def __init__(self, root, transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(root, transform=transform,
                                      target_transform=target_transform)


        self.data = []
        self.targets = []

        for f in sorted(self.files):
            self.data.append(np.load('{}/{}'.format(root, f)))
            self.targets.append(np.load('{}/{}'.format(root, self.labels)))
        self.data = np.vstack(self.data).reshape(-1, 32, 32, 3)
        self.targets = np.vstack(self.targets).reshape(-1).astype(np.int)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = CIFAR10C(root='/home/nutszebra/Downloads/CIFAR-10-C')
