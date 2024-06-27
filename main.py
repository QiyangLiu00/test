import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch import optim
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from datasets.DataSet import MyCustomDataset
from models import get_model
from optimizers.adam_lr_scheduler import Adam_LRScheduler, AdamLRScheduler
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime


def main(device, args):
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=True, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model).to(device)
    # model = torch.nn.DataParallel(model)
    model_dict = model.state_dict()
    print(model)

    # # define optimizer
    # optimizer = get_optimizer(
    #     args.train.optimizer.name, model,
    #     lr=args.train.base_lr * args.train.batch_size / 256,
    #     momentum=args.train.optimizer.momentum,
    #     weight_decay=args.train.optimizer.weight_decay)
    #
    # lr_scheduler = LR_Scheduler(
    #     optimizer,
    #     args.train.warmup_epochs, args.train.warmup_lr * args.train.batch_size / 256,
    #     args.train.num_epochs, args.train.base_lr * args.train.batch_size / 256,
    #                               args.train.final_lr * args.train.batch_size / 256,
    #     len(train_loader),
    #     constant_predictor_lr=True  # see the end of section 4.2 predictor
    # )
    optimizer = optim.Adam(model.parameters(), lr=args.train.base_lr * args.train.batch_size / 256)
    lr_scheduler = AdamLRScheduler(optimizer)
    # lr_scheduler = Adam_LRScheduler(optimizer, args.train.base_lr)

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        args.train.epoch = epoch
        if epoch % 20 == 0:
            args.train.positives = args.train.positives + 1
            print(args.train.positives)

        model.train()

        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2), labels, img_location) in enumerate(local_progress):
            model.zero_grad()

            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True), args, img_location)
            loss = data_dict['loss'].mean()  # ddp
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr': lr_scheduler.get_lr()})

            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
            accuracy = knn_monitor(model.backbone, memory_loader, test_loader, device,
                                   k=min(args.train.knn_k, len(memory_loader.dataset)),
                                   hide_progress=args.hide_progress)

        epoch_dict = {"epoch": epoch, "accuracy": accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
        gc.collect()
        torch.cuda.empty_cache()

    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir,
                              f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth")
    # datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')

    if args.eval is not False:
        args.eval_from = model_path
        linear_eval(args)


if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)

    # path = './logs/in-progress_0308095718_simsiam-cifar10-experiment-resnet18_cifar_variant1'
    # completed_log_dir = path.replace('in-progress', 'completed')
    # os.rename(path, completed_log_dir)

    print(f'Log file has been saved to {completed_log_dir}')
