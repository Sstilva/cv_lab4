import os
import torch
import numpy as np
from dataset import lowlight_dataset
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS

def get_metrics(enh_img: torch.Tensor, gt_img: torch.Tensor, device: str='cpu'):
    '''
        Функция для измерения метрик. Расчитывает среднее значение по сэмплу
        Arguments:
            :enh_img: изображения после обработки
            :gt_img: ground-truth изображения
            :device: cpu/cuda
    '''
    # значение каждого пикселя должно быть в интервале от 0.0 до 1.0
    enh_img = enh_img.to(device)/255 if enh_img.max() > 1 else enh_img.to(device)
    gt_img = gt_img.to(device)/255 if gt_img.max() > 1 else gt_img.to(device)
    # PSNR
    psnr = PSNR(data_range=1.0).to(device)
    psnr_score = psnr(enh_img, gt_img).item()
    # SSIM
    ms_ssim = SSIM(data_range=1.0).to(device)
    ssim_score = ms_ssim(enh_img, gt_img).item()
    # LILPS
    lpips = LPIPS('alex').to(device)
    lpips_score = lpips(enh_img, gt_img).item()
    return psnr_score, ssim_score, lpips_score


def evaluate(batch_size=8, device='cpu'):
    '''Оценка метрик решения'''
    
    img_names = os.listdir('dataset/low')
    dataset = lowlight_dataset(img_names, 'dataset')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    psnr_list = np.array([])
    ssim_list = np.array([])
    lpips_list = np.array([])
    times = np.array([])

    for lowlight_img, gt_img in dataloader:

        # Улучшение качества изображения
        enhanced_image = lowlight_img
        ################################

        psnr_score, ssim_score, lpips_score = get_metrics(enhanced_image, gt_img, device=device)

        psnr_list = np.append(psnr_list, psnr_score)
        ssim_list = np.append(ssim_list, ssim_score)
        lpips_list = np.append(lpips_list, lpips_score)

    print(f'PSNR: {psnr_list.mean()}\nSSIM: {ssim_list.mean()}\nLPIPS: {lpips_list.mean()}')
