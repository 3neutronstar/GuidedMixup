import numpy as np
import torch


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img:torch.tensor):
        """ 
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if img.dim==4:
            n = img.size(0)
            c = img.size(1)
            h = img.size(2)
            w = img.size(3)
            #print(h)
            #print(w)
            mask = np.ones((n, c, h, w), np.float32)

            for i in range(n):
                for hole in range(self.n_holes):
                    y = np.random.randint(h)
                    x = np.random.randint(w)

                    y1 = np.clip(y - self.length // 2, 0, h)
                    y2 = np.clip(y + self.length // 2, 0, h)
                    x1 = np.clip(x - self.length // 2, 0, w)
                    x2 = np.clip(x + self.length // 2, 0, w)

                    mask[i, :, y1:y2, x1:x2] = 0.

            mask = torch.from_numpy(mask).to('cuda')
            mask = mask.expand_as(img)
            img = img * mask
        elif img.dim==3:
            c = img.size(1)
            h = img.size(2)
            w = img.size(3)
            mask = np.ones((c, h, w), np.float32)

            for hole in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[:, y1:y2, x1:x2] = 0.

            mask = torch.from_numpy(mask).to('cuda')
            mask = mask.expand_as(img)
            img = img * mask

        return img