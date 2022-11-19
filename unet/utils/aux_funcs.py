import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from functools import partial
import yaml
import logging
import logging.config
import pickle as pkl
import argparse
import pathlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
from utils.data_utils import get_data_loaders
from configs.general_configs import (
    MODEL_CONFIGS_DIR_PATH,

    OUTPUT_DIR,

    EPOCHS,
    BATCH_SIZE,
    OPTIMIZER_LR,

    OPTIMIZER,
    OPTIMIZER_WEIGHT_DECAY,
    OPTIMIZER_MOMENTUM_DECAY,

    KERNEL_REGULARIZER_TYPE,
    KERNEL_REGULARIZER_L1,
    KERNEL_REGULARIZER_L2,
    KERNEL_REGULARIZER_FACTOR,
    KERNEL_REGULARIZER_MODE,

    AWS_INPUT_BUCKET_NAME,
    AWS_INPUT_BUCKET_SUBDIR,
    AWS_INPUT_REGION,

    AWS_OUTPUT_BUCKET_NAME,
    AWS_OUTPUT_BUCKET_SUBDIR,
    AWS_OUTPUT_REGION,
    CHECKPOINT_DIR,
    OPTIMIZER_BETA_1,
    OPTIMIZER_BETA_2,
    CHECKPOINT_FILE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IN_CHANNELS,
    OUT_CHANNELS,
    VAL_PROP,
    OPTIMIZER_RHO,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_DAMPENING,
    OPTIMIZER_LR_DECAY,
    OPTIMIZER_EPS,
    TRAIN_DATA_DIR,
    INFERENCE_DATA_DIR,
    TEST_DATA_DIR,
    NUM_WORKERS,
    REDUCE_LR_ON_PLATEAU_FACTOR,
    PIN_MEMORY,
    REDUCE_LR_ON_PLATEAU_MIN, INFO_BAR_HEIGHT
)
from utils.logging_funcs import info_log

__author__ = 'mchlsdrv@gmail.com'

plt.style.use('seaborn')


# MISCELLANEOUS
def read_yaml(data_file: pathlib.Path):
    data = None
    if data_file.is_file():
        with data_file.open(mode='r') as f:
            data = yaml.safe_load(f.read())
    return data


def get_filename(file: str):
    return file[::-1][file[::-1].index('.') + 1:][::-1]


def to_pickle(file, name: str, save_dir: str or pathlib.Path):
    os.makedirs(save_dir, exist_ok=True)

    pkl.dump(file, (save_dir / (name + '.pkl')).open(mode='wb'))


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--gpu_id', type=int, choices=[gpu_id for gpu_id in range(0, torch.cuda.device_count())], default=-1 if torch.cuda.device_count() > 0 else -1, help='The ID of the GPU (if there is any) to run the network on (e.g., --gpu_id 1 will run the network on GPU #1 etc.)')

    parser.add_argument('--train_continue', default=False, action='store_true', help=f'If to continue the training from the checkpoint saved at \'{CHECKPOINT_DIR}\'')
    parser.add_argument('--train_data_dir', type=str, default=TRAIN_DATA_DIR, help='The path to the directory where the images and corresponding masks are stored')
    parser.add_argument('--test_data_dir', type=str, default=TEST_DATA_DIR, help='The path to the directory where the test images and corresponding masks are stored')
    parser.add_argument('--infer_local', default=False, action='store_true', help=f'If to use local images for inference')
    parser.add_argument('--inference_image_dir', type=str, default=INFERENCE_DATA_DIR, help='The path to the directory where the inference images and corresponding masks are stored')

    parser.add_argument('--aws_input_bucket_name', type=str, default=AWS_INPUT_BUCKET_NAME, help=f'Path to the bucket where the images are downloaded from')
    parser.add_argument('--aws_input_region', type=str, default=AWS_INPUT_REGION, help=f'The region of the input bucket')
    parser.add_argument('--aws_input_bucket_subdir', type=str, default=AWS_INPUT_BUCKET_SUBDIR, help=f'The subdirectory where the images are located')

    parser.add_argument('--aws_output_bucket_name', type=str, default=AWS_OUTPUT_BUCKET_NAME, help=f'Path to the bucket where the images are pushed')
    parser.add_argument('--aws_output_region', type=str, default=AWS_OUTPUT_REGION, help=f'The region of output client')
    parser.add_argument('--aws_output_bucket_subdir', type=str, default=AWS_OUTPUT_BUCKET_SUBDIR, help=f'The subdirectory where the images are located')

    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    parser.add_argument('--image_width', type=int, default=IMAGE_WIDTH, help='The width of the images that will be used for network training and inference. If not specified, will be set to IMAGE_WIDTH as in general_configs.py file.')
    parser.add_argument('--image_height', type=int, default=IMAGE_HEIGHT, help='The height of the images that will be used for network training and inference. If not specified, will be set to IMAGE_HEIGHT as in general_configs.py file.')

    parser.add_argument('--in_channels', type=int, default=IN_CHANNELS, help='The number of channels in an input image (e.g., 3 for RGB, 1 for Grayscale etc)')
    parser.add_argument('--out_channels', type=int, default=OUT_CHANNELS, help='The number of channels in the output image (e.g., 3 for RGB, 1 for Grayscale etc)')

    parser.add_argument('--load_model', default=False, action='store_true', help=f'If to load the model')
    parser.add_argument('--model_configs_dir', type=str, default=MODEL_CONFIGS_DIR_PATH, help='The path to the directory where the configuration of the network are stored (in YAML format)')

    # - TRAINING
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='The number of workers to load the data')
    parser.add_argument('--val_prop', type=float, default=VAL_PROP, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--checkpoint_file', type=str, default=CHECKPOINT_FILE, help=f'The path to the file which contains the checkpoints of the model')

    # - OPTIMIZERS
    # optimizer
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw', 'sparse_adam', 'nadam', 'adadelta', 'adamax', 'adagrad'], default=OPTIMIZER,  help=f'The optimizer to use')

    parser.add_argument('--optimizer_lr', type=float, default=OPTIMIZER_LR, help=f'The initial learning rate of the optimizer')
    parser.add_argument('--optimizer_lr_decay', type=float, default=OPTIMIZER_LR_DECAY, help=f'The learning rate decay for Adagrad optimizer')
    parser.add_argument('--optimizer_beta_1', type=float, default=OPTIMIZER_BETA_1, help=f'The exponential decay rate for the 1st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_beta_2', type=float, default=OPTIMIZER_BETA_2, help=f'The exponential decay rate for the 2st moment estimates (Adam, Nadam, Adamax)')
    parser.add_argument('--optimizer_rho', type=float, default=OPTIMIZER_RHO, help=f'The decay rate (Adadelta, RMSprop)')
    parser.add_argument('--optimizer_amsgrad', default=False, action='store_true', help=f'If to use the Amsgrad function (Adam, Nadam, Adamax)')

    parser.add_argument('--optimizer_weight_decay', type=float, default=OPTIMIZER_WEIGHT_DECAY, help=f'The weight decay for ADAM, NADAM')
    parser.add_argument('--optimizer_momentum', type=float, default=OPTIMIZER_MOMENTUM, help=f'The momentum for SGD')
    parser.add_argument('--optimizer_dampening', type=float, default=OPTIMIZER_DAMPENING, help=f'The dampening for momentum')
    parser.add_argument('--optimizer_momentum_decay', type=float, default=OPTIMIZER_MOMENTUM_DECAY, help=f'The momentum for NADAM')
    parser.add_argument('--optimizer_nesterov', default=False, action='store_true', help=f'If to use the Nesterov momentum (SGD)')
    parser.add_argument('--optimizer_centered', default=False, action='store_true', help=f'If True, gradients are normalized by the estimated variance of the gradient; if False, by the un-centered second moment. Setting this to True may help with training, but is slightly more expensive in terms of computation and memory. (RMSprop)')

    parser.add_argument('--no_drop_block', default=False, action='store_true', help=f'If to use the drop_block in the network')
    parser.add_argument('--drop_block_keep_prob', type=float, help=f'The probability to keep the block')
    parser.add_argument('--drop_block_block_size', type=int, help=f'The size of the block to drop')

    parser.add_argument('--kernel_regularizer_type', type=str, choices=['l1', 'l2', 'l1l2'], default=KERNEL_REGULARIZER_TYPE, help=f'The type of the regularization')
    parser.add_argument('--kernel_regularizer_l1', type=float, default=KERNEL_REGULARIZER_L1, help=f'The strength of the L1 regularization')
    parser.add_argument('--kernel_regularizer_l2', type=float, default=KERNEL_REGULARIZER_L2, help=f'The strength of the L2 regularization')
    parser.add_argument('--kernel_regularizer_factor', type=float, default=KERNEL_REGULARIZER_FACTOR, help=f'The strength of the orthogonal regularization')
    parser.add_argument('--kernel_regularizer_mode', type=str, choices=['rows', 'columns'], default=KERNEL_REGULARIZER_MODE, help=f"The mode ('columns' or 'rows') of the orthogonal regularization")

    # - CALLBACKS
    parser.add_argument('--wandb', default=False, action='store_true', help=f'If to use the Weights and Biases board')

    parser.add_argument('--early_stopping_patience', type=int, default=-1, help=f'The number of epochs to wait until the training will stop')

    parser.add_argument('--reduce_lr_on_plateau_patience', type=int, default=-1, help=f'The number of epochs to wait until the learning rate will be reduced by reduce_lr_on_plateau_factor')
    parser.add_argument('--reduce_lr_on_plateau_factor', type=float, default=REDUCE_LR_ON_PLATEAU_FACTOR, help=f'The factor to reduce the learning rate by')

    return parser


def get_runtime(seconds: float):
    hrs = int(seconds // 3600)
    mins = int((seconds - hrs * 3600) // 60)
    sec = seconds - hrs * 3600 - mins * 60

    # - Format the strings
    hrs_str = str(hrs)
    if hrs < 10:
        hrs_str = '0' + hrs_str
    min_str = str(mins)
    if mins < 10:
        min_str = '0' + min_str
    sec_str = f'{sec:.3}'
    if sec < 10:
        sec_str = '0' + sec_str

    return hrs_str + ':' + min_str + ':' + sec_str + '[H:M:S]'


# MODEL / AUGS / OPTIMIZER / DEVICE
def get_augs(args: dict):
    train_augs = A.Compose([
        A.Resize(height=args.get('image_height'), width=args.get('image_width')),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.CLAHE(p=1.),
        A.ToFloat(p=1.),
        ToTensorV2()
    ])

    val_augs = A.Compose([
        A.Resize(height=args.get('image_height'), width=args.get('image_width')),
        A.CLAHE(p=1.),
        A.ToFloat(p=1.),
        ToTensorV2()
    ])

    return train_augs, val_augs


def get_accuracy(data_loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device=device)
            y = y.to(device='cpu').unsqueeze(1)

            preds = torch.sigmoid(model(x)).to('cpu')
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            acc = (num_correct / num_pixels)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    dice_score = dice_score / len(data_loader)
    print(f'> Accuracy: {acc * 100 :.4f}%')
    print(f'> Dice Score: {dice_score:.4f}')

    model.train()

    return acc, dice_score


def save_preds(data_loader, model, save_dir: str or pathlib.Path, device='cuda'):
    model.eval()
    for idx, (x, y) in enumerate(data_loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{save_dir}/pred_mask_{idx}.png')
        torchvision.utils.save_image(y.float().unsqueeze(1), f'{save_dir}/gt_mask_{idx}.png')

    model.train()


def get_model_configs(model_name: str, configs_dir: pathlib.Path, logger: logging.Logger):
    model_configs = read_yaml(configs_dir / (model_name + '_configs.yml'))
    if model_configs is not None:
        info_log(logger=logger, message=f'The model configs for \'{model_name}\' were loaded from \'{configs_dir}\'')
    else:
        info_log(logger=logger, message=f'No model configs were found for model \'{model_name}\'')
    return model_configs


def get_optimizer(params, algorithm: str, args: dict):
    optimizer = None
    if algorithm == 'sgd':
        optimizer = partial(
            torch.optim.SGD,
            params=params,
            momentum=args.get('momentum'),
            weight_decay=args.get('weight_decay'),
            nesterov=args.get('nesterov'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adam':
        optimizer = partial(
            torch.optim.Adam,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adamw':
        optimizer = partial(
            torch.optim.AdamW,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            amsgrad=args.get('amsgrad'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'sparse_adam':
        optimizer = partial(
            torch.optim.SparseAdam,
            params=params,
            betas=args.get('betas'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'nadam':
        optimizer = partial(
            torch.optim.NAdam,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            momentum_decay=args.get('momentum_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adamax':
        optimizer = partial(
            torch.optim.Adamax,
            params=params,
            betas=args.get('betas'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adadelta':
        optimizer = partial(
            torch.optim.Adadelta,
            params=params,
            rho=args.get('rho'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    elif algorithm == 'adagrad':
        optimizer = partial(
            torch.optim.Adadelta,
            params=params,
            lr_decay=args.get('lr_decay'),
            weight_decay=args.get('weight_decay'),
            eps=OPTIMIZER_EPS,
        )
    return optimizer(lr=args.get('lr'))


def get_device(gpu_id: int = 0, logger: logging.Logger = None):
    n_gpus = torch.cuda.device_count()

    print(f'> Number of available GPUs: {n_gpus}')
    device = 'cpu'
    if n_gpus > 0:
        try:
            if -1 < gpu_id < n_gpus:
                print(f'> Setting GPU to: {gpu_id}')

                device = f'cuda:{gpu_id}'

                print(f'''
    ======================
    = Running on: {device} =
    ======================
                ''')
            elif gpu_id > n_gpus - 1:

                device = f'cuda'
                print(f'''
    =====================================
    = Running on all the available GPUs =
    =====================================
                    ''')

        except RuntimeError as err:
            if isinstance(logger, logging.Logger):
                logger.exception(err)

    return device


def save_checkpoint(state, filename: str or pathlib.Path):
    print(f'=> Saving checkpoint to \'{filename}\' ...')
    torch.save(state, filename)


def load_checkpoint(model, checkpoint_file):
    print('=> Loading checkpoint ...')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])


# - TRAIN / VAL / TEST UTILS
def train_fn(model, data_loader, optimizer, loss_fn, scaler, device: str, verbose: bool = False):
    # - TRAIN
    loop = tqdm(data_loader)

    if verbose:
        print('\n> Training ...')

    train_losses = np.array([])
    for btch_idx, (data, trgts) in enumerate(loop):
        data = data.to(device=device)
        trgts = trgts.float().unsqueeze(1).to(device=device)
        # trgts = trgts.type(torch.LongTensor).to(device=device)

        # Forward pass
        with torch.cuda.amp.autocast():
            preds = model(data)
            loss = loss_fn(preds, trgts)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())
        train_losses = np.append(train_losses, loss.item())

    return train_losses.mean()


def val_fn(model, data_loader, loss_fn, device: str, verbose: bool = False):
    # - VALIDATION
    if verbose:
        print('\n> Validating ...')
    model.eval()
    val_loop = tqdm(data_loader)
    val_losses = np.array([])
    for btch_idx, (data, trgts) in enumerate(val_loop):
        data = data.to(device=device)
        trgts = trgts.float().unsqueeze(1).to(device=device)
        # trgts = trgts.type(torch.LongTensor).to(device=device)

        with torch.no_grad():
            # Forward pass
            with torch.cuda.amp.autocast():
                preds = model(data)
                loss = loss_fn(preds, trgts).item()

        # Update tqdm loop
        val_loop.set_postfix(loss=loss)
        val_losses = np.append(val_losses, loss)

    model.train()

    return val_losses.mean()


def detection_fn(image, model, device: str):
    # - VALIDATION
    model.eval()
    img = image.to(device=device)

    with torch.no_grad():
        # Forward pass
        with torch.cuda.amp.autocast():
            preds = model(img)

    model.train()

    return preds.to('cpu')


def train_model(model, loss_fn, args, augs: dict, callbacks: dict, device: str, save_dir: pathlib.Path, logger: logging.Logger = None):
    # - Print some examples
    train_dir = save_dir / 'train'
    os.makedirs(save_dir, exist_ok=True)

    plots_dir = train_dir / 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    val_preds_dir = train_dir / 'val_preds'
    os.makedirs(val_preds_dir, exist_ok=True)

    chkpt_dir = save_dir / 'checkpoints'
    os.makedirs(chkpt_dir, exist_ok=True)

    if args.load_model:
        chkpt_fl = pathlib.Path(args.checkpoint_file)
        if chkpt_fl.is_file():
            load_checkpoint(torch.load(chkpt_fl), model)

    optimizer = get_optimizer(
        params=model.parameters(),
        algorithm=args.optimizer,
        args=dict(
            lr=args.optimizer_lr,
            lr_decay=args.optimizer_lr_decay,
            betas=(args.optimizer_beta_1, args.optimizer_beta_2),
            weight_decay=args.optimizer_weight_decay,
            momentum=args.optimizer_momentum,
            momentum_decay=args.optimizer_momentum_decay,
            dampening=args.optimizer_dampening,
            rho=args.optimizer_rho,
            nesterov=args.optimizer_nesterov,
            amsgrad=args.optimizer_amsgrad
        )
    )

    train_data_loader, val_data_loader = get_data_loaders(
        data_dir=args.train_data_dir,
        batch_size=args.batch_size,
        train_augs=augs.get('train_augs'),
        val_augs=augs.get('val_augs'),
        val_prop=args.val_prop,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        logger=logger
    )

    # - Train loop
    print(f'''
    ====================
    == TRAINING MODEL ==
    ====================
    ''')
    best_loss = np.inf
    no_imprv_epchs = 0
    train_losses = np.array([])
    val_losses = np.array([])
    accs = np.array([])
    dices = np.array([])
    scaler = torch.cuda.amp.GradScaler()
    for epch in range(args.epochs):
        print(f'\n== Epoch: {epch + 1}/{args.epochs} ({100*(epch + 1)/args.epochs:.2f}% done) ==')

        train_loss = train_fn(model=model, data_loader=train_data_loader, optimizer=optimizer, loss_fn=loss_fn, scaler=scaler, device=device, verbose=True)
        train_losses = np.append(train_losses, train_loss)

        val_loss = val_fn(model=model, data_loader=val_data_loader, loss_fn=loss_fn, device=device, verbose=True)
        val_losses = np.append(val_losses, val_loss)

        # - Save the best model
        if val_loss < best_loss:

            # - Save checkpoint
            print(f'<!!> val_loss improved from {best_loss:.3f} -> {val_loss:.3f}')

            # - Update the best loss
            best_loss = val_loss

            checkpoint = dict(
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            save_checkpoint(state=checkpoint, filename=chkpt_dir / f'best_val_loss_chkpt.pth.tar')

            # - Reset the non-improvement counter
            no_imprv_epchs = 0

        else:
            print(f'<!> val_loss ({val_loss:.3f}) did not improved from the last best_loss value ({best_loss:.3f})')

            # - Increase the non-improvement counter
            no_imprv_epchs += 1

        # - Check accuracy
        acc, dice = get_accuracy(data_loader=val_data_loader, model=model, device=device)
        accs = np.append(accs, acc)
        dices = np.append(dices, dice)

        # - Callbacks
        # > Plot samples
        fig, ax = plt.subplots()
        ax.plot(np.arange(epch+1), train_losses, label='Train')
        ax.plot(np.arange(epch+1), val_losses, label='Validation')
        ax.plot(np.arange(epch+1), accs, label='Accuracy')
        ax.plot(np.arange(epch+1), dices, label='Dice Score')

        plt.legend()
        plt.savefig(f'{plots_dir}/train_stats.png')

        save_preds(data_loader=val_data_loader, model=model, save_dir=val_preds_dir, device=device)

        # > Early Stopping
        early_stp_clbk = callbacks.get('early_stopping')
        if early_stp_clbk.get('use'):
            patience = early_stp_clbk.get('patience')
            if no_imprv_epchs >= patience:
                print(f'<x> No improvement was recorded for {patience} epochs - stopping the training!')
                break

        # > LR Reduction on Plateau
        reduce_lr_on_plt_clbk = callbacks.get('reduce_lr_on_plateau')
        if reduce_lr_on_plt_clbk.get('use'):
            patience = reduce_lr_on_plt_clbk.get('patience')
            if no_imprv_epchs >= patience:
                lr = optimizer.param_groups[0]['lr']
                fctr = reduce_lr_on_plt_clbk.get('factor')
                new_lr = fctr * lr
                if new_lr < REDUCE_LR_ON_PLATEAU_MIN:
                    print(f'<x> The lr ({new_lr:.3f}) was reduced beyond its smallest possible value ({REDUCE_LR_ON_PLATEAU_MIN:.3f}) - stopping the training!')
                    break

                optimizer.param_groups[0]['lr'] = new_lr

                print(f'<!> No improvement was recorded for {patience} epochs - reducing lr by factor {fctr:.3f}, from {lr:.3f} -> {new_lr:.3f}!')

    return model


def test_model(model, loss_fn, args, augs, device: str, save_dir: pathlib.Path, logger: logging.Logger = None):

    test_data_loader, _ = get_data_loaders(
        data_dir=args.test_data_dir,
        batch_size=args.batch_size,
        train_augs=augs,
        val_augs=None,
        val_prop=0.,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        logger=logger
    )

    print(f'''
    ====================
    == TESTING MODEL ==
    ====================
    ''')
    val_loss = val_fn(data_loader=test_data_loader, model=model, loss_fn=loss_fn, device=device)

    # - Check accuracy
    acc, dice = get_accuracy(data_loader=test_data_loader, model=model, device=device)

    print(f'''
    =================================
    ========= TEST RESULTS ==========
    =================================
    > Loss: {val_loss:.4f}
    > Accuracy: {100*acc:.4f}%
    > Dice Score: {dice:.4f}
    =================================
    ''')

    # - Print some examples
    save_dir = save_dir / 'test/preds'
    os.makedirs(save_dir, exist_ok=True)
    save_preds(data_loader=test_data_loader, model=model, save_dir=save_dir, device=device)


def detect_images(image_dir: pathlib.Path, model, augs, device: str, save_dir: pathlib.Path):
    os.makedirs(save_dir, exist_ok=True)

    img_fls = os.listdir(image_dir)
    print(f'\nDetecting images: \n{img_fls}\n ...')

    for img_fl in tqdm(img_fls):
        img = cv2.imread(str(image_dir / img_fl), -1)[:-INFO_BAR_HEIGHT, :]
        aug_res = augs(image=img)
        img = aug_res.get('image').unsqueeze(0)

        print(f'\nDetecting image \'{img_fl}\' ...')
        img_dets = detection_fn(
            image=img,
            model=model,
            device=device
        )

        masked_img = img * img_dets

        # - Get file name
        img_nm = get_filename(file=img_fl)
        torchvision.utils.save_image(masked_img, f'{save_dir}/{img_nm}.png')
