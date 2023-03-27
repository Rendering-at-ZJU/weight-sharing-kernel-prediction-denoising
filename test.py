import os
import numpy as np
import pyexr

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import dataset
import net

def BMFRGammaCorrection(img):
    if isinstance(img, np.ndarray):
        return np.clip(np.power(np.maximum(img, 0.0), 0.454545), 0.0, 1.0)
    elif isinstance(img, torch.Tensor):
        return torch.pow(torch.clamp(img, min=0.0, max=1.0), 0.454545)

def ComputeMetrics(truth_img, test_img):    
    truth_img = BMFRGammaCorrection(truth_img)
    test_img  = BMFRGammaCorrection(test_img)
    
    SSIM = structural_similarity(truth_img, test_img, multichannel=True)
    PSNR = peak_signal_noise_ratio(truth_img, test_img)
    return SSIM, PSNR

def Inference(model, device, dataloader, saving_root=""):
    model.eval()
    SSIMs = []
    PSNRs = []
    with torch.no_grad():
        for img_idx, (inputs_crops, targets_crops) in enumerate(dataloader):
            inputs = inputs_crops.to(device, non_blocking=True)
            targets = targets_crops.to(device, non_blocking=True)
            outputs = model(inputs).detach()
            
            output = outputs.cpu().numpy()[0].transpose((1, 2, 0)) # BMFR
            target = targets.cpu().numpy()[0].transpose((1, 2, 0))
            SSIM, PSNR = ComputeMetrics(target, output)
            SSIMs.append(SSIM)
            PSNRs.append(PSNR)
                
            pyexr.write(os.path.join(saving_root, str(img_idx)+".exr"), output)
            
    print("Test:")
    SSIM_mean = np.mean(SSIMs)
    PSNR_mean = np.mean(PSNRs)
    print("mean SSIM:", SSIM_mean)
    print("mean PSNR:", PSNR_mean)
    SSIMs.append("mean: "+str(SSIM_mean))
    PSNRs.append("mean: "+str(PSNR_mean))
    np.savetxt(os.path.join(saving_root, "ssim.txt"), SSIMs, fmt="%s")
    np.savetxt(os.path.join(saving_root, "psnr.txt"), PSNRs, fmt="%s")
            
    return SSIM_mean, PSNR_mean



if __name__ == "__main__":
    torch.cuda.set_device(1)
    torch.backends.cudnn.deterministic = True  # same result for cpu and gpu
    torch.backends.cudnn.benchmark = False # key in here: Should be False. Ture will make the training process unstable
    device = torch.device("cuda")

    database = dataset.DataBase()
    dataset_test = dataset.BMFRFullResAlDataset(database, use_test=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    timestamp = "example-test"
    episode_name = "classroom"
    test_saving_root = os.path.join("results", timestamp, episode_name)
    os.makedirs(test_saving_root, exist_ok=True)

    model_pretrain = net.repWeightSharingKPNet(device).to(device)
    map_location = {'cuda:0': 'cuda:1'}
    checkpoint = torch.load("checkpoints/classroom/model.pt", map_location) # NOTE
    model_pretrain.load_state_dict(checkpoint['model_state_dict'])
    model_deployment = net.repWeightSharingKPNet(device, is_deployment=True, model_pretrain=model_pretrain).to(device)
    Inference(model_deployment, device, dataloader_test, test_saving_root)
