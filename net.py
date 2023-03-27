import numpy as np
import torch
import torch.nn as nn
import CUSTOM_WSKP_auto

def BMFRGammaCorrection(img):
    if isinstance(img, np.ndarray):
        return np.clip(np.power(np.maximum(img, 0.0), 0.454545), 0.0, 1.0)
    elif isinstance(img, torch.Tensor):
        return torch.pow(torch.clamp(img, min=0.0, max=1.0), 0.454545)

def SumKernelParameter(kernel_size, device):
    kernel = np.ones((1, 1, kernel_size, kernel_size), dtype=np.float32)
    kernel = torch.from_numpy(kernel.reshape(1, 1, kernel_size, kernel_size))
   
    kernel = kernel.to(device)
    return torch.nn.Parameter(kernel, requires_grad=False)

def make_equivalent_kernel(conv5, conv3, conv1, has_identity_pass=False):
    kernel = conv5.weight + torch.nn.functional.pad(conv3.weight, [1, 1, 1, 1]) + torch.nn.functional.pad(conv1.weight, [2, 2, 2, 2])
    if has_identity_pass:
        identity_pass_weight = torch.eye(kernel.shape[0], device=conv5.weight.device).reshape((kernel.shape[0], kernel.shape[0], 1, 1))
        identity_pass_weight = torch.nn.functional.pad(identity_pass_weight , [2, 2, 2, 2])
        kernel += identity_pass_weight
    conv = nn.Conv2d(in_channels=kernel.shape[1], out_channels=kernel.shape[0], kernel_size=5, stride=1, padding=2, bias=False)
    conv.weight.data = kernel.detach().cpu().clone()
    return conv

class repWeightSharingKPNet(nn.Module):
    def __init__(self, device, is_deployment=False, model_pretrain=None):
        super(repWeightSharingKPNet, self).__init__()
        self.device = device
        self.is_deployment = is_deployment
        
        conv1_in_channels = 10
        self.base_depth = 14
                    
        self.kernel_num = 6
        self.padding_size = self.kernel_num
        self.kernel_base_size = 3
        self.kernel_size_stride = 2
        
        self.convSs = []
        for i in range(self.kernel_num):
            kernel_size = self.kernel_base_size + i * self.kernel_size_stride
            padding = (kernel_size - 1) // 2
            convS = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
            convS.weight = SumKernelParameter(kernel_size=kernel_size, device=device)
            self.convSs.append(convS)
        
        if not self.is_deployment:
            self.conv1_5 = nn.Conv2d(in_channels=conv1_in_channels, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.conv1_3 = nn.Conv2d(in_channels=conv1_in_channels, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_1 = nn.Conv2d(in_channels=conv1_in_channels, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.lrelu1 = nn.LeakyReLU(0.1)
            
            self.conv2_5 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.conv2_3 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2_1 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False) 
            self.lrelu2 = nn.LeakyReLU(0.1)
            
            self.conv3_5 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.conv3_3 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3_1 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.lrelu3 = nn.LeakyReLU(0.1)
            
            self.conv4_5 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.conv4_3 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv4_1 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.lrelu4 = nn.LeakyReLU(0.1)
            
            self.conv5_5 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=5, stride=1, padding=2, bias=False)
            self.conv5_3 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv5_1 = nn.Conv2d(in_channels=self.base_depth, out_channels=self.base_depth, kernel_size=1, stride=1, padding=0, bias=False)
            self.lrelu5 = nn.LeakyReLU(0.1)
            
            output_channels = self.kernel_num + self.kernel_num
            self.conv_final_5 = nn.Conv2d(in_channels=self.base_depth, out_channels=output_channels, kernel_size=5, stride=1, padding=2, bias=False)
            self.conv_final_3 = nn.Conv2d(in_channels=self.base_depth, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_final_1 = nn.Conv2d(in_channels=self.base_depth, out_channels=output_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.softmax = nn.Softmax2d()
        else:
            self.conv1 = make_equivalent_kernel(model_pretrain.conv1_5, model_pretrain.conv1_3, model_pretrain.conv1_1)
            self.lrelu1 = nn.LeakyReLU(0.1)

            self.conv2 = make_equivalent_kernel(model_pretrain.conv2_5, model_pretrain.conv2_3, model_pretrain.conv2_1, has_identity_pass=True)
            self.lrelu2 = nn.LeakyReLU(0.1)

            self.conv3 = make_equivalent_kernel(model_pretrain.conv3_5, model_pretrain.conv3_3, model_pretrain.conv3_1, has_identity_pass=True)
            self.lrelu3 = nn.LeakyReLU(0.1)

            self.conv4 = make_equivalent_kernel(model_pretrain.conv4_5, model_pretrain.conv4_3, model_pretrain.conv4_1, has_identity_pass=True)
            self.lrelu4 = nn.LeakyReLU(0.1)

            self.conv5 = make_equivalent_kernel(model_pretrain.conv5_5, model_pretrain.conv5_3, model_pretrain.conv5_1, has_identity_pass=True)
            self.lrelu5 = nn.LeakyReLU(0.1)

            self.conv_final = make_equivalent_kernel(model_pretrain.conv_final_5, model_pretrain.conv_final_3, model_pretrain.conv_final_1)
            self.softmax = nn.Softmax2d()


      
        
    def forward(self, x_in):
        x_irradiance = x_in[:, 0:3]
        x_albedo = x_in[:, 3:6]
        x_inputs = torch.cat((BMFRGammaCorrection(x_irradiance * x_albedo),
                                            x_in[:, 3:]), axis=1)
        
        x0_out = x_inputs
        
        if not self.is_deployment:
            x1_5 = self.conv1_5(x0_out)
            x1_3 = self.conv1_3(x0_out)
            x1_1 = self.conv1_1(x0_out)
            x1_out = self.lrelu1(x1_5 + x1_3 + x1_1)

            x2_5 = self.conv2_5(x1_out)
            x2_3 = self.conv2_3(x1_out)
            x2_1 = self.conv2_1(x1_out)
            x2_out = self.lrelu2(x2_5 + x2_3 + x2_1 + x1_out)
            
            x3_5 = self.conv3_5(x2_out)
            x3_3 = self.conv3_3(x2_out)
            x3_1 = self.conv3_1(x2_out)
            x3_out = self.lrelu3(x3_5 + x3_3 + x3_1 + x2_out)
            
            x4_5 = self.conv4_5(x3_out)
            x4_3 = self.conv4_3(x3_out)
            x4_1 = self.conv4_1(x3_out)
            x4_out = self.lrelu4(x4_5 + x4_3 + x4_1 + x3_out)
            
            x5_5 = self.conv5_5(x4_out)
            x5_3 = self.conv5_3(x4_out)
            x5_1 = self.conv5_1(x4_out)
            x5_out = self.lrelu5(x5_5 + x5_3 + x5_1 + x4_out)
            
            x_final_in = x5_out
            x_final_5 = self.conv_final_5(x_final_in)
            x_final_3 = self.conv_final_3(x_final_in)
            x_final_1 = self.conv_final_1(x_final_in)
            x_final_out = x_final_5 + x_final_3 + x_final_1

            x_guidemap = torch.exp(x_final_out[:, :self.kernel_num]) # from 20-NV-AdaptiveSampling
            x_alpha = self.softmax(x_final_out[:, self.kernel_num:])

            B, _, H, W = x_inputs.shape
            # Every channel apply 2d filter with same kernel weight,
            # from https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840
            x_out = 0.0
            for i in range(self.kernel_num):
                x_guidemap_windowsum = self.convSs[i](x_guidemap[:, i:(i+1)])
                x_out += x_alpha[:, i:i+1] * (self.convSs[i]((x_guidemap[:, i:(i+1)] * x_irradiance).view(-1, 1, H, W)).view(B, -1, H, W) / x_guidemap_windowsum)


            x_out = x_out * x_albedo

        else:
            x1_out = self.lrelu1(self.conv1(x0_out))
            x2_out = self.lrelu2(self.conv2(x1_out))
            x3_out = self.lrelu3(self.conv3(x2_out))
            x4_out = self.lrelu4(self.conv4(x3_out))
            x5_out = self.lrelu5(self.conv5(x4_out))
            x_final_out = self.conv_final(x5_out)

            x_guidemap = torch.exp(x_final_out[:, :self.kernel_num]) # from 20-NV-AdaptiveSampling
            x_alpha = self.softmax(x_final_out[:, self.kernel_num:])

            # use padding to avoid the boundary judgement in reconstruciton cuda kernel
            x_guidemap = torch.nn.functional.pad(x_guidemap, (self.padding_size, self.padding_size, self.padding_size, self.padding_size))
            x_alpha = torch.nn.functional.pad(x_alpha, (self.padding_size, self.padding_size, self.padding_size, self.padding_size))
            x_irradiance = torch.nn.functional.pad(x_irradiance, (self.padding_size, self.padding_size, self.padding_size, self.padding_size))
            x_albedo = torch.nn.functional.pad(x_albedo, (self.padding_size, self.padding_size, self.padding_size, self.padding_size))
            x_out = CUSTOM_WSKP_auto.forward(x_irradiance, x_albedo, x_guidemap, x_alpha)
            x_out = x_out[:, :, self.padding_size:-self.padding_size, self.padding_size:-self.padding_size]
        return x_out