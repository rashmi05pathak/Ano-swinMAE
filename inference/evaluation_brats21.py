#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from functools import partial
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp
from utils.pos_embed import get_2d_sincos_pos_embed


# In[6]:


import swin_mae_inference_wind


# In[7]:


import sys
import glob
import torch
sys.path.insert(0,'/storage/ayantika/Ayantika/analyse_1/Rashmi/')
import slice_data_h5 as sdl_h5
from monai.transforms import (
    Orientationd, EnsureChannelFirstd, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
    # ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
)


# In[8]:


############## Data loader 
transforms1 = Compose(
    [
     EnsureChannelFirstd(('image','label')),
     Orientationd(('image','label'),'RAS'),
#      Spacingd(('image','label'),(1,1,1)),        
     Resized(keys = ('image'),spatial_size = (224, 224,-1),mode = 'trilinear' ,align_corners = True),
     Resized(keys = ('label'),spatial_size = (224, 224,-1),mode = 'nearest' ),
     ScaleIntensityD(('image',)),
     ToTensord(('image','label')),
    ]
)

# path_nii = '/storage/Ayantika/Data_final/ixi_raw/IXI_preprocessed_Data/reg_n4/**T2.nii.gz'
# path_h5 = '/storage/Ayantika/analyse_1/Rashmi/brain_cache/h5_data/IXI/T2'

# path_nii = '/storage/Ayantika/analyse_1/BRATS2020/brats_training_data/data_raw/**/**t2.nii'
# path_h5 = '/storage/Ayantika/analyse_1/Rashmi/brain_cache/h5_data/Brats20_T2'

path_nii = '/storage/ayantika/Ayantika/Data_final/brats21/**/**t2.nii.gz'
path_h5 = '/storage/ayantika/Ayantika/analyse_1/Rashmi/brain_cache/h5_data/Brats21_T2'

# path_nii = '/storage/Ayantika/Data_final/ixi_raw/IXI_preprocessed_Data/T1/**T1.nii.gz'
# path_h5 = '/storage/Ayantika/analyse_1/Rashmi/brain_cache/h5_data/IXI/T1'

nslices_per_image = 155
start_slice= 75 #60
end_slice= 60 #45  
trainlist_1 = [{'image':x} for x in glob.glob(path_nii)]
datalist =  trainlist_1
mask_nii = '/storage/ayantika/Ayantika/Data_final/brats21/**/**seg.nii.gz'
masklist = [{'label':x} for x in glob.glob(mask_nii)]
 ### The loader is such that it would create h5 files if they are not created when the loader is called and executed                                                               
h5cacheds = sdl_h5.H5CachedDataset(datalist,masklist,transforms1,h5cachedir = path_h5,\
                                   nslices_per_image = nslices_per_image,\
                                   start_slice = start_slice,\
                                   end_slice = end_slice)
torch.multiprocessing.set_sharing_strategy('file_system')
#sampler_train = torch.utils.data.RandomSampler(datalist)
test_loader = torch.utils.data.DataLoader(h5cacheds,batch_size = 1,shuffle = True)


# In[9]:


def coordinates_to_patch_indexes(x, y, patch_size=16, image_size=224):
    """
    Convert image coordinates into set of patch indexes.

    Parameters:
    - x, y: Coordinates in the image.
    - patch_size: Size of each patch (default is 16).
    - image_size: Size of the image (default is 224).

    Returns:
    - patch_index_x, patch_index_y: Patch indexes corresponding to the given coordinates.
    """
    patch_index_x = x // patch_size
    patch_index_y = y // patch_size

    # Ensure that the coordinates are within the image boundaries
    patch_index_x = min(max(patch_index_x, 0), image_size // patch_size - 1)
    patch_index_y = min(max(patch_index_y, 0), image_size // patch_size - 1)

    return patch_index_x, patch_index_y

def run_one_image(img,window_arr, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    
    # run MAE
    loss, y, mask = model(x.float(),window_arr)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
#     # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, 16 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', x)
    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
#*************Commenting the code for temporary computation*************************
#     #make the plt figure larger
#     plt.rcParams['figure.figsize'] = [24, 24]

#     plt.subplot(1, 4, 1)
#     plt.imshow(x[0], 'gray')

#     plt.subplot(1, 4, 2)
#     plt.imshow(im_masked[0], 'gray')

#     plt.subplot(1, 4, 3)
#     plt.imshow(y[0], 'gray')

# #     plt.subplot(1, 5, 4)
# #     plt.imshow(im_paste[0], 'gray')
    
#     plt.subplot(1, 4, 4)
#     plt.imshow((x[0]-y[0]), 'gray')

#     plt.show()
    return y[0]

#ChatGPT Automatic code for section 2.Methodology/Localization of paper https://arxiv.org/pdf/2010.01942.pdf 
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def sliding_window(image, window_size, step_size):
    for y in range(48, image.shape[0] - window_size[0] + 1, step_size):
        for x in range(25, image.shape[1] - window_size[1] + 1, step_size):
            yield (x, y, image[y:y + window_size[0], x:x + window_size[1]])

def minmax_normalization(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = 255 * ((image - min_val) / (max_val - min_val))
    return normalized_image.astype(np.uint8)

def reconstruction_loss(original, reconstructed):
    return np.abs(original[:,:,0] - reconstructed[:,:,0])
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2gray
from skimage import io
from skimage import exposure
from skimage.measure import regionprops
def swin_mae_evaluation(image,label,model,gamma=32,k=32):
    
    # chkpt_dir = r'saved_model/epoch_14728.pt'
    # model_mae = prepare_model(chkpt_dir, 'swin_mae')
    # print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    #*******we are running the model for each window****************************####
    window_list = []
    #*************************Addition of code by Rashmi*************************************#
    # Example usage
    heatmap = np.zeros_like(image[:, :, 0], dtype=float)
    #heatmap = torch.zeros((image_height - window_height + 1, image_width - window_width + 1))
    for (x, y, window) in sliding_window(image, (gamma, gamma), k):#Hint: x,y denotes the origin of the window i.e. start of the window
        window_list.append(window)
        # Perform minmax normalization on original and reconstructed windows
        normalized_image = minmax_normalization(image)
        normalized_window_orig = minmax_normalization(window)
        # Example usage:
        patch_indexes = []
        # Ensure that the coordinates are within the image boundaries
        for a in range(x,x+gamma):
            for b in range(y,y+gamma):
                x_coordinate = a
                y_coordinate = b
                patch_index_x, patch_index_y = coordinates_to_patch_indexes(x_coordinate, y_coordinate)
                final_patch_number = (patch_index_x)*14+(patch_index_y)
                patch_indexes.append(final_patch_number)
                #rint(f"Coordinates ({x_coordinate}, {y_coordinate}) correspond to patch indexes ({patch_index_x}, {patch_index_y}).")
        arr = np.unique(patch_indexes)
        window_arr = arr.tolist()
        reconstructed_image = run_one_image(image,window_arr, model)
            # run MAE
        reconstructed_window = reconstructed_image[y:y+gamma, x:x+gamma]
        normalized_window_recon = minmax_normalization(reconstructed_window.numpy()) #reconstructed image normalization
        # Compute L1 reconstruction loss
        loss = reconstruction_loss(normalized_window_orig, normalized_window_recon)
#         plt.imshow(loss)
        # Accumulate loss in the corresponding region of the heatmap
        heatmap[y:y + gamma, x:x + gamma] += loss

    # Normalize the heatmap values to [0, 1]
    combined_heatmap = heatmap / np.max(heatmap)

    return combined_heatmap



# In[10]:


model = swin_mae_inference_wind.SwinMAE()
# model.load_state_dict(torch.load('/storage/Ayantika/analyse_1/Rashmi/Swin-MAE/output_dir_T2/checkpoint-250.pth')['model'])
model.load_state_dict(torch.load('/storage/ayantika/Ayantika/analyse_1/Rashmi/Swin-MAE/output_dir_T2/checkpoint-250.pth')['model'])


# In[11]:


import cv2

from sklearn.metrics import precision_recall_curve, average_precision_score

def compute_auprc(y_pred, y):
    y_pred = y_pred.flatten()
    y = y.flatten()
    precisions, recalls, thresholds = precision_recall_curve(y.astype(int), y_pred)
    auprc = average_precision_score(y.astype(int), y_pred)
    return auprc, precisions, recalls, thresholds

def calculate_dice_score(gt, pred):
    """
    Calculate Dice score from true positives (TP), false positives (FP), and false negatives (FN).

    Parameters:
    - tp: True positives
    - fp: False positives
    - fn: False negatives

    Returns:
    - Dice score
    """
    tp, fp, fn = calculate_confusion_matrix(gt, pred)
    dice_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return dice_score,tp, fp, fn
def calculate_confusion_matrix(gt, pred):
    # True Positive (TP): Intersection of Ground Truth and Predicted
    tp = np.sum((gt == 1) & (pred == 1))

    # False Positive (FP): Predicted - True Positive
    fp = np.sum((gt == 0) & (pred == 1))

    # False Negative (FN): Ground Truth - True Positive
    fn = np.sum((gt == 1) & (pred == 0))

    return tp, fp, fn

def visualize_confusion_map(gt, pred):
    # Calculate confusion matrix
    tp, fp, fn = calculate_confusion_matrix(gt, pred)

    # Create masks for TP, FP, and FN
    tp_mask = (gt == 1) & (pred == 1)
    fp_mask = (gt == 0) & (pred == 1)
    fn_mask = (gt == 1) & (pred == 0)

    # Create a combined image with different colors for TP, FP, and FN
    combined_image = np.zeros_like(gt, dtype=np.uint8)
    combined_image[tp_mask] = 255  # White for True Positive
    combined_image[fp_mask] = 128  # Gray for False Positive
    combined_image[fn_mask] = 50   # Dark Yellow for False Negative

    # Display the images
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap='gray')
    plt.title('Ground Truth')

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap='gray')
    plt.title('Predicted')

    plt.subplot(1, 3, 3)
    plt.imshow(combined_image, cmap='viridis')  # Use viridis colormap for combined image
    plt.title(f'TP: {tp}, FP: {fp}, FN: {fn}')

    plt.show()
    
    
def post_process(comb_heat_map,org_img,gt,viz=False):   
    org_img = org_img[:,:,0]       
#     comb_heat_map[:,150:] = 1
    
#     kernel = np.ones((5, 5), np.uint8)
#     eroded_image = cv2.morphologyEx(((org_img>0)*255).astype('uint8'), cv2.MORPH_ERODE, kernel)
#     plt.imshow(eroded_image,cmap='gray')
    comb_heat_map = comb_heat_map * (org_img>0.1)
    

    if viz:
        plt.imshow(comb_heat_map,cmap = 'gray')
        plt.colorbar()
        plt.title('comb_heat_map')
        plt.show()
        
        plt.imshow(gt, cmap = 'gray')
        plt.colorbar()
        plt.title('gt')
        plt.show()
        plt.imshow(org_img, cmap = 'gray')
        plt.colorbar()
        plt.title('org_img')
        plt.show()

    ############################################################

    heat_map_rev = (1 - comb_heat_map) * (org_img>0.1)
    plt.imshow(heat_map_rev,cmap='gray')
    kernel = np.ones((1,1), np.uint8)
    eroded_image = cv2.morphologyEx((heat_map_rev*255).astype('uint8'), cv2.MORPH_ERODE, kernel)
    eroded_image= (eroded_image/255)>0.5
    # plt.figure()
    # plt.imshow(eroded_image,cmap='gray')
    # plt.colorbar()
    from skimage.segmentation import felzenszwalb
    from skimage.measure import regionprops
    segments_old = felzenszwalb(comb_heat_map, scale=75, sigma=0.8, min_size=100)

    segments = eroded_image*segments_old
    # Get properties of superpixel regions
    region_props = regionprops(segments, intensity_image=comb_heat_map)
#                                org_img)
    if viz:
#         plt.figure()
        plt.title('eroded_image')
        plt.imshow( eroded_image,cmap='gray')
        plt.show()
#     sorted_regions = sorted(region_props, key=lambda prop: prop.area, reverse=True)
#     # Extract the top four regions
#     top_regions_ = sorted_regions[:10]

#     intensity_sorted_regions = sorted(top_regions_, key=lambda prop: prop.intensity_mean, reverse=True)
#     top_regions = intensity_sorted_regions[:10]
    intensity_sorted_regions = sorted(region_props, key=lambda prop: prop.intensity_mean, reverse=True)
    top_regions = intensity_sorted_regions[:7]




    # plt.imshow(segments_old,cmap='gray')
    # plt.imshow(segments,cmap='gray')
    predicted_mask_comb = np.zeros_like(gt)
    for ii in range(len(top_regions)):
        predicted_mask_comb = predicted_mask_comb + (segments == top_regions[ii].label)

    if viz:
#         plt.figure()
        plt.title('predicted_mask_comb')
        plt.imshow( predicted_mask_comb,cmap='gray')
        plt.show()
#         plt.figure()
#         plt.imshow(gt,cmap='gray')




    # predicted_mask_final = (segments_old == (np.unique((predicted_mask_comb*segments_old))[-1]))

    kernel = np.ones((5, 5), np.uint8)
    predicted_mask_comb = cv2.morphologyEx(predicted_mask_comb.astype('uint8'), cv2.MORPH_DILATE, kernel)
    #         visualize_confusion_map(gt, predicted_mask_comb)



    import ants
    a=org_img
    ants_image = ants.from_numpy(a)
    img_ = ants.resample_image(ants_image, (224,224), 1, 0)
    mask = ants.get_mask(img_)
    img_seg = ants.atropos(a=img_, m='[0.2,1x1]', c='[2,0]', 
                           i='kmeans[4]', x=mask)
    img_seg = img_seg['segmentation'].numpy()

    kernel = np.ones((20, 20), np.uint8)
    eroded_mask = cv2.morphologyEx(((org_img>0.1)*255).astype('uint8'), cv2.MORPH_ERODE, kernel)
    if viz:
        plt.title('atropos')
        plt.imshow(img_seg)
        plt.show()
    #         if np.sum(img_seg==3)>=np.sum(img_seg==4):
    #             choose_ind = 2

    #         elif np.sum(img_seg==4)>np.sum(img_seg==3):
    choose_ind = 2

    img_seg_final = ((img_seg>=2) & (img_seg<4)) *eroded_mask
    # plt.figure()
    # plt.imshow(img_seg_final,cmap='gray')

    kernel = np.ones((1, 1), np.uint8)
    eroded_img_seg_final = cv2.morphologyEx(img_seg_final.astype('uint8'), cv2.MORPH_DILATE, kernel)
    segments_org_img = felzenszwalb(eroded_img_seg_final, scale=75, sigma=0.1, min_size=10)
    # plt.figure()
    # plt.imshow(segments_org_img,cmap='gray')

    region_props_org_img = regionprops(segments_org_img, intensity_image=org_img)





    extracted_img = np.zeros_like(predicted_mask_comb)
    for u in np.unique(segments_org_img):
        if u != 0:

            if np.sum(predicted_mask_comb * (segments_org_img==u))>0:
                extracted_img = (extracted_img+ (segments_org_img==u))

    #         if viz:
#     visualize_confusion_map(gt, extracted_img)
    t2=time.time()
    print("Time Taken",t2-t1)
    
    return gt, extracted_img 


# In[12]:


for i, batch_ in enumerate(test_loader):
    print("i",i)
    batch1 = torch.einsum('nchw->nhwc', batch_['image'])
    stacked_img = np.stack((batch1[:,:,:,0],)*3, axis=-1)
    break


# In[13]:


#Testing with window size 64 by 64 with pixel shift 64
import time
dice_coeff = 0.0
brats_good_dice = []
brats_bad_dice = []
dice_sum = 0
dict_comb = {}
count = 0
for i, batch_ in enumerate(test_loader):
    t1 = time.time()
    print("i",i)
    batch1 = torch.einsum('nchw->nhwc', batch_['image'])
    stacked_img = np.stack((batch1[:,:,:,0],)*3, axis=-1)
    j = 0
    if(np.sum(batch_['label'][j,0,:,:].numpy())>1024): #sending only mri slices with bigger pathologies
        
#         if i==5:
#             if j==2:
        combined_heatmap = swin_mae_evaluation(stacked_img[j],batch_['label'][j,0,:,:],model,64,64)
        print('combined_heatmap',combined_heatmap.shape)
        gt, extracted_img = post_process(combined_heatmap,stacked_img[j],batch_['label'][j,0].numpy(),viz=False)
        dice_score,tp, fp, fn = calculate_dice_score(gt, extracted_img)
        auprc_score,_,_,_ = compute_auprc(gt, extracted_img)
        print(dice_score,auprc_score,tp, fp, fn)

        dict_comb.update({count:{"filename":batch_['filepath'][j],\
                             "gt_name":batch_['slicenum'][j],\
                             "combined_heatmap":combined_heatmap,\
                             "org_img":stacked_img[j],\
                             "gt":batch_['label'][j,0].numpy(),\
                             "dice_score":dice_score,\
                             "auprc_score":auprc_score,\
                             "tp":tp,\
                             "fp":fp, \
                             "fn":fn}})
        count +=1          


# In[24]:


len(dict_comb)
# dict_keys(['map_name', 'gt_name', 'img_name', 'dice_score', 'tp', 'fp', 'fn'])


# In[25]:


len(test_loader)


# In[26]:


dice_accum_without_zero = []
for key in dict_comb:
        dice_accum_without_zero.append(dict_comb[key]['dice_score'])
#     break


# In[27]:


np.mean(dice_accum_without_zero)


# In[20]:


dice_accum_ = []
for key in dict_comb:
        dice_accum_.append(dict_comb[key]['dice_score'])
#     break


# In[21]:


np.mean(dice_accum_)


# In[28]:


auprc_accum = []
for key in dict_comb:
    auprc_accum.append(dict_comb[key]['auprc_score'])


# In[29]:


sum(auprc_accum)/len(auprc_accum)


# In[18]:


get_ipython().system('nvidia-smi')


# In[ ]:




