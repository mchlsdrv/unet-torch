import os
import pathlib
import time
from datetime import datetime


from configs.general_configs import (
    LOSS_FN,
    CONFIGS_DIR_PATH,
)
from custom.model import UNet
from utils.aux_funcs import (
    get_model_configs,
    get_device,
    get_runtime
)
from utils.aux_funcs import (
    get_augs,
    get_arg_parser,
    train_model,
    test_model
)

__author__ = 'mchlsdrv@gmail.com'

from utils.logging_funcs import get_logger


if __name__ == '__main__':
    # GENERAL
    # - Get the argument parser
    parser = get_arg_parser()
    args = parser.parse_args()

    # - Create the directory for the current procedure
    # Time stamp
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    current_run_dir = pathlib.Path(args.output_dir) / f'{ts}'
    os.makedirs(current_run_dir, exist_ok=True)

    # - Configure the logger
    logger = get_logger(
        configs_file=CONFIGS_DIR_PATH / 'logger_configs.yml',
        save_file=current_run_dir / f'logs.log'
    )

    # - Choose device to run on
    device = get_device(gpu_id=args.gpu_id, logger=logger)
    # print(f'device: {device}')

    train_augs, val_augs = get_augs(
        args=dict(
            image_width=args.image_width,
            image_height=args.image_height,
        )
    )

    model_configs = get_model_configs(model_name='unet', configs_dir=args.model_configs_dir, logger=logger)
    model = UNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        n_features=model_configs.get('n_features'),
        output_dir=args.output_dir,
        logger=logger
    ).to(device)

    # - TRAIN MODEL
    t_start = time.time()
    model = train_model(
        model=model,
        loss_fn=LOSS_FN,
        args=args,
        augs=dict(
            train_augs=train_augs,
            val_augs=val_augs
        ),
        callbacks=dict(
            early_stopping=dict(
                use=True if args.early_stopping_patience > 0 else False,
                patience=args.early_stopping_patience
            ),
            reduce_lr_on_plateau=dict(
                use=True if args.reduce_lr_on_plateau_patience > 0 else False,
                patience=args.reduce_lr_on_plateau_patience,
                factor=args.reduce_lr_on_plateau_factor
            )
        ),
        device=device,
        save_dir=current_run_dir,
        logger=logger
    )

    # - TEST MODEL
    test_model(
        model=model,
        loss_fn=LOSS_FN,
        args=args,
        augs=val_augs,
        device=device,
        save_dir=current_run_dir,
        logger=logger
    )

    t_end = time.time() - t_start

    print('Total runtime: ', get_runtime(seconds=t_end))
