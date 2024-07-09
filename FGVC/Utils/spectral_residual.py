import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Utils.augmentation import Identity
import torchvision.transforms.functional as TF

def series_filter(values, kernel_size=3):
    """
    Filter a time series. Practically, calculated mean value inside kernel size.
    As math formula, see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html.
    :param values:
    :param kernel_size:
    :return: The list of filtered average
    """
    filter_values = torch.cumsum(values,dim=[2,3], dtype=torch.float)

    filter_values[kernel_size:] = filter_values[kernel_size:] - filter_values[:-kernel_size]
    filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

    for i in range(1, kernel_size):
        filter_values[i] /= i + 1

    return filter_values

class SpectralResidual(object):
    def __init__(self,blur=7,sigma=3, kernel_size=3,device='cuda'):
        self.kernel_size=kernel_size
        self.boxfilter=nn.Conv2d(1,1,self.kernel_size,bias=False,padding=int((kernel_size-1)/2),padding_mode='replicate')
        self.boxfilter.weight= nn.Parameter(torch.ones((1,1,self.kernel_size,self.kernel_size),dtype=torch.float)/float(self.kernel_size*self.kernel_size),requires_grad=False)
        self.boxfilter=self.boxfilter.to(device)
        self.to_gray=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
        ])
        if blur==0:
            self.blur=Identity()
        else:
            self.blur=transforms.GaussianBlur((blur,blur),(sigma,sigma))
        self.eps=1e-10
        self.resize=transforms.Resize((128,128))

    def transform_saliency_map(self, values):
        """
        Transform a time-series into spectral residual, which is method in computer vision.
        For example, See https://github.com/uoip/SpectralResidualSaliency.
        :param values: a list or numpy array of float values.
        :return: silency map and spectral residual
        """
        values=self.to_gray(values)
        if values.shape[2] > 128:
            values=self.resize(values)

        freq = torch.fft.fft2(values)
        mag = (freq.real ** 2 + freq.imag ** 2+self.eps).sqrt()
        spectral_residual = (mag.log() - self.boxfilter(mag.log())).exp()

        freq.real = freq.real * spectral_residual / mag
        freq.imag = freq.imag * spectral_residual / mag

        saliency_map = torch.fft.ifft2(freq)
        saliency_map.squeeze_(1)
        return saliency_map

    def transform_spectral_residual(self, values):
        with torch.no_grad():
            saliency_map = self.transform_saliency_map(values)
            spectral_residual = (saliency_map.real ** 2 + saliency_map.imag ** 2).sqrt()
            spectral_residual=self.blur(spectral_residual)
            if values.shape[2] > 128:
                spectral_residual=TF.resize(spectral_residual,size=(values.shape[2],values.shape[3]))
            spectral_residual.squeeze_(1)
            
        # spectral_residual=(spectral_residual-spectral_residual.min())/(spectral_residual.max()-spectral_residual.min())

        return spectral_residual

if __name__=='__main__':
    saliency=SpectralResidual()
    a=torch.randn((1,3,32,32))
    b=saliency.transform_saliency_map(a)
    print(b.size(),'hi')