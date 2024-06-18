import argparse
import json
import numpy as np
import os
from pathlib import Path
import glob
import time

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import swin_mae
from utils.engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # common parameters
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--checkpoint_encoder', default='', type=str)
    parser.add_argument('--checkpoint_decoder', default='', type=str)
    #parser.add_argument('--data_path', default=r'/storage/Ayantika/Data_final/IXI_colin/reg_n4_skl_strp/train', type=str)  # fill in the dataset path here
    parser.add_argument('--mask_ratio', default=0.35, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # model parameters
    parser.add_argument('--model', default='swin_mae', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # optimizer parameters
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # other parameters
    parser.add_argument('--output_dir', default='./output_dir_T1',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_T1',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    #parser.add_argument('--pin_mem', action='store_true',
                        #help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    #parser.set_defaults(pin_mem=True)

    return parser


import sys
sys.path.insert(0,'/storage/Ayantika/analyse_1/Rashmi/')
import slice_data_h5 as sdl_h5
from monai.transforms import (
    Orientationd, AddChanneld, Compose, ToTensord, Spacingd,Resized,ScaleIntensityD,ResizeWithPadOrCropd
    # ScaleIntensityD, ScaleIntensityRangeD, AdjustContrastD, RandAffined, ToNumpyd,RepeatChannelD
)


def get_input(iterator, train_loader):
    try:
        inputs = next(iterator)
    except StopIteration:
        # Create new generator if the previous generator is exhausted.
        iterator = iter(train_loader)
        inputs = next(iterator)
        #     input_images, labels = input
        #print(inputs)
    input_images= inputs['image']
    labels = inputs['label']

    return input_images, labels


class AverageMeter(object):
    '''
    compute and store the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import math


# +
def main(args):
        # Fixed random seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set up training equipment
    device = torch.device("cuda")
    cudnn.benchmark = True

    # Defining data augmentation
#     transform_train = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()
#     ])

    #     # Set dataset
    #     dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     data_loader_train = torch.utils.data.DataLoader(
    #         dataset_train, sampler=sampler_train,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=True
    #     )

    ############## Data loader 
    transforms1 = Compose(
        [
         AddChanneld(('image','label')),
         Orientationd(('image','label'),'RAS'),
    #      Spacingd(('image','label'),(1,1,1)),        
         Resized(keys = ('image'),spatial_size = (224, 224,-1),mode = 'trilinear' ,align_corners = True),
         Resized(keys = ('label'),spatial_size = (224, 224,-1),mode = 'nearest' ),
         ScaleIntensityD(('image',)),
         ToTensord(('image','label')),
        ]
    )
    path_nii = '/storage/Ayantika/Data_final/ixi_raw/IXI_preprocessed_Data/T1/**T1.nii.gz'
    path_h5 = '/storage/Ayantika/analyse_1/Rashmi/brain_cache/h5_data/IXI/T1'
    nslices_per_image = 155
    start_slice= 60
    end_slice= 45  
    IXI_datapath_1 = path_nii
    trainlist_1 = [{'image':x} for x in glob.glob(IXI_datapath_1)]
    datalist =  trainlist_1

    mask_nii = '/storage/Ayantika/Data_final/ixi_raw/IXI_preprocessed_Data/mask/**T1_mask.nii.gz'
    masklist = [{'label':x} for x in glob.glob(mask_nii)]
     ### The loader is such that it would create h5 files if they are not created when the loader is called and executed                                                               
    h5cacheds = sdl_h5.H5CachedDataset(datalist,masklist,transforms1,h5cachedir = path_h5,\
                                       nslices_per_image = nslices_per_image,\
                                       start_slice = start_slice,\
                                       end_slice = end_slice)
    torch.multiprocessing.set_sharing_strategy('file_system')
    #sampler_train = torch.utils.data.RandomSampler(datalist)
    train_loader = torch.utils.data.DataLoader(h5cacheds,batch_size = 10,shuffle = True)
    # Log output
    # if args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = SummaryWriter()
    # else:
    #     log_writer = None
    model_dir = '/storage/Ayantika/analyse_1/Rashmi/Swin-MAE/saved_model_T1'
    save_dir = '/storage/Ayantika/analyse_1/Rashmi/Swin-MAE/output_dir_T1/tb'
    log_writer = SummaryWriter(log_dir=save_dir) 
    # Set model
    model = swin_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, mask_ratio=args.mask_ratio)
    model_without_ddp = model
#**************************************
#     args.weight_decay = 5e-2
#     args.momentum = (0.9, 0.95)
#     args.epochs_warmup = 40
#     args.warmup_from = 1e-4
#     args.lr_decay_rate = 1e-2
#     eta_min = args.lr * (args.lr_decay_rate ** 3)
#     args.warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.epochs_warmup / args.epochs)) / 2
#*******************************************
#     args.print_freq = 5
#     args.save_freq = 100
#     args.label_smoothing = True
#     args.smoothing = 0.1
    #learning rate scheduler: warmup + consine
#     def lr_lambda(epoch):
#         if epoch < args.epochs_warmup:
#             p = epoch / args.warmup_epochs
#             lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
#         else:
#             eta_min = args.lr * (args.lr_decay_rate ** 3)
#             lr = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
#         return lr
    # Set optimizer
    param_groups = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=5e-2, betas=(0.9, 0.95)) 
    loss_scaler = NativeScaler()
   # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    #Create model
    state_dict = torch.load('/storage/Ayantika/analyse_1/Rashmi/Swin-MAE/swin_tiny_patch4_window7_224.pth')
    swin_unet_t = state_dict['model']
    key_swin = list(swin_unet_t.keys())
    for i,key_mae in enumerate(model_without_ddp.state_dict()):
        if key_mae in key_swin:
            model_without_ddp.state_dict()[key_mae] = swin_unet_t[key_mae]
#     model_without_ddp.load_state_dict(torch.load('/storage/Ayantika/analyse_1/Rashmi/Swin-MAE/output_dir_T1/checkpoint-85.pth')['model'])
    model_without_ddp.to(device)
    misc.load_model(args=args, model_without_ddp=model_without_ddp)
#*******************This logic is written by the original codebase itself********************#
    # Start the training process
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model_without_ddp, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch + 1)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
#     train_iterator = iter(train_loader)
#    #Training or Finetuning code for MAE model
#     for epoch in range(1, args.epochs + 1):
#         # records
#         ts = time.time()
#         losses = AverageMeter()
#      # records
#         images, targets = get_input(train_iterator, train_loader)
#         images, targets = images.float(), targets.float().to(torch.device("cuda"))
#         batch1 = torch.einsum('nchw->nhwc', images)
#         stacked_img = np.stack((batch1[:,:,:,0],)*3, axis=-1)
#         stacked_img = torch.einsum('nhwc->nchw', torch.from_numpy(stacked_img))
#         # run MAE
#         loss, _, _ = model(stacked_img.to(torch.device("cuda")))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         # record
#         losses.update(loss.item(), args.batch_size)
#         print(" Finished epoch:{} AVG Training Loss:{:.3f} ".format(epoch+1,losses.avg,))

#         # log
#     #     tb_logger.log_value('loss', losses.avg, epoch)
#         print(stacked_img.shape)
#         grid_batch_input = ((torchvision.utils.make_grid(stacked_img[0:10,:,:,:])))
#         log_writer.add_image(f"input_image/{epoch}", grid_batch_input,epoch)

# #         grid_batch_output = ((torchvision.utils.make_grid(y1[0:10,:,:,:])))
# #         writer.add_image(f'output_image/{epoch}', grid_batch_output ,epoch)

#         # print
#         if epoch % args.print_freq == 0:
#             print('- epoch {:3d}, time, {:.2f}s, loss {:.4f}'.format(epoch, time.time() - ts, losses.avg))

#         # save checkpoint
#         if losses.avg <= 0.005:
#             torch.save(model.state_dict(), os.path.join(model_dir, 'epoch_'+str(epoch)+'.pt'))
#             #save_file = os.path.join(args.ckpt_folder, 'epoch_{:d}.ckpt'.format(epoch))
#             #save_ckpt(model, optimizer, args, epoch, save_file=save_file)

#     # writer.flush()
#     log_writer.close()


# -

if __name__ == '__main__':
    arg = get_args_parser()
    arg = arg.parse_args()
    if arg.output_dir:
        Path(arg.output_dir).mkdir(parents=True, exist_ok=True)
    main(arg)

    





        











