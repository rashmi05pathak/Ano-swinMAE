import math
import sys

import torch
import numpy as np
import torchvision

import utils.misc as misc
import utils.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #print(samples['image'])
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        input_image = samples['image']
        batch1 = torch.einsum('nchw->nhwc', input_image)
        stacked_img = np.stack((batch1[:,:,:,0],)*3, axis=-1)
        stacked_img = torch.einsum('nhwc->nchw', torch.from_numpy(stacked_img)).to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, pred, mask = model(stacked_img)
            y1 = model.unpatchify(pred)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    #After completion of all steps in one epoch, i.e. in last step we will have one entry 
    grid_batch_input = ((torchvision.utils.make_grid(stacked_img[0:10,:,:,:])))
    log_writer.add_image(f"input_image/{epoch}", grid_batch_input,epoch)
    grid_batch_output = ((torchvision.utils.make_grid(y1[0:10,:,:,:])))
    log_writer.add_image(f'output_image/{epoch}', grid_batch_output ,epoch)
    ####log_writer.add_image(f'output_image', grid_batch_output ,epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
