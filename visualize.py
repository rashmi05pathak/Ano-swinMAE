import sys
import os
import numpy as np

import torch
import matplotlib.pyplot as plt
import nibabel as nib

import swin_mae

sys.path.append('..')


# define the utils
def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    image = torch.clip(image * 255, 0, 255).int()
    image = np.asarray(Image.fromarray(np.uint8(image)).resize((224, 448)))
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir_, arch='swin_mae'):
    # build model
    model = getattr(swin_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir_, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(x, model):
    x = torch.tensor(x)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float())
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)
    y = y * mask

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [12, 6]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)

    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()


if __name__ == '__main__':
    path = r'/storage/Ayantika/Data_final/IXI_colin/reg_n4_skl_strp/test/IXI299-Guys-0893-T2.nii.gz'
    #img_name = r''
    img = nib.load(path).get_fdata()
    img = img.resize((224, 224))
    img = np.asarray(img) / 255.
    #assert img.shape == (224, 224, 3)
    batch1 = torch.einsum('nchw->nhwc', images)
    stacked_img = np.stack((batch1[:,:,:,0],)*3, axis=-1)
    stacked_img = torch.einsum('nhwc->nchw', torch.from_numpy(stacked_img))
    # run MAE

    chkpt_dir = r'/Ayantika/analyse_1/Rashmi/Swin-MAE/saved_model/epoch_1600.pt'
    model_mae = prepare_model(chkpt_dir, 'swin_mae')
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    run_one_image(stacked_img, model_mae)
