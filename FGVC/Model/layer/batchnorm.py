import torch
import torch.nn as nn
import torch.nn.functional as F


"""
special thanks to https://github.com/ptrblck/pytorch_misc/tree/master
"""
class CustomBatchNorm2d_depreciated(nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        condition = x.sum(dim=[2,3])
        if (condition == 0).sum()==0:
            return super().forward(x)
        else:
            num_channels=x.shape[1]
            # new_tensor=torch.zeros_like(x)
            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:  # type: ignore
                    self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            r"""
            Decide whether the mini-batch stats should be used for normalization rather than the buffers.
            Mini-batch stats are used in training mode, and in eval mode when buffers are None.
            """
            if self.training:
                bn_training = True
            else:
                bn_training = (self.running_mean is None) and (self.running_var is None)

            r"""
            Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
            passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
            used for normalization (i.e. in eval mode when buffers are not None).
            """
            assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
            assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
              
            if self.training and self.track_running_stats:
                for c in range(num_channels): # conduct batch norm for each channel
                    
                    if condition[:,c].sum()==0: # if the sum of the channel is 0, then skip
                        continue
                    else:
                        with torch.no_grad():
                            mask= torch.nonzero(condition[:,c])
                        todo_normalize=x[:,c,:,:].index_select(0,mask.squeeze(1)).unsqueeze(1) # select the non-zero elements
                    mask=mask.squeeze(1)
                    normalized_channel = F.batch_norm(
                        todo_normalize,
                        # If buffers are not to be tracked, ensure that they won't be updated
                        self.running_mean[c] if not self.training or self.track_running_stats else None,
                        self.running_var[c] if not self.training or self.track_running_stats else None,
                        self.weight[c], self.bias[c], bn_training, exponential_average_factor, self.eps)
                    x[:,c,:,:]=x[:,c,:,:].index_copy(0,mask,normalized_channel.squeeze(1))
                    # new_tensor[:,c,:,:].index_copy_(0,mask,normalized_channel.squeeze(1))

                    # print(new_tensor[:,c,:,:].shape,normalized_channel.squeeze(1).shape)
                    # new_tensor[:,c,:,:].scatter_add_(0,mask.view(-1,1,1).detach(),normalized_channel.squeeze(1))
            return x


class CustomBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(CustomBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        B,C,H,W=input.shape
        with torch.no_grad():
            condition = (input.sum(dim=[2,3],keepdim=True) != 0).float()
        if condition.mean()==1: # no zero across the channel
            return super().forward(input)
        else:
            with torch.no_grad():
                expanded_condition=condition.expand_as(input)
                N=expanded_condition.sum() # number of non-zero elements -> activated number of values in the feature map
                num_valid_C= N / ( H*W*B) 
            if num_valid_C==1:
                return super().forward(input)
                
            self._check_input_dim(input)

            exponential_average_factor = 0.0

            if self.training and self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum


            # calculate running estimates
            if self.training:
                mean = input.sum([0, 2, 3]) / N *num_valid_C

                valid_C=(mean!=0)

                # use biased var in train
                var = (input - mean[None, :, None, None]).pow(2).sum([0, 2, 3]) / ( int(N / num_valid_C)) # unbiased
                n = input.numel() / input.size(1)
                with torch.no_grad():
                    self.running_mean = (exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * self.running_mean) * valid_C.float() + self.running_mean * (~valid_C).float() # update only the valid channels
                    # update running_var with unbiased var
                    self.running_var = (exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var ) * valid_C.float() + self.running_var * (~valid_C).float() # update only the valid channels
            else:
                mean = self.running_mean
                var = self.running_var

            input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
            if self.affine:
                input = (input * self.weight[None, :, None, None] + self.bias[None, :, None, None]) * expanded_condition

            return input
