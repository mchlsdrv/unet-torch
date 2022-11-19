import os
import pathlib
from datetime import datetime

from configs.general_configs import CONFIGS_DIR_PATH, DELETE_ON_FINISH
from custom.model import UNet
from utils.aux_funcs import (
    get_device, get_model_configs, get_augs, load_checkpoint, detect_images
)
import matplotlib.pyplot as plt
from utils.aux_funcs import get_arg_parser
from utils.cloud_utils import S3Utils
from utils.logging_funcs import get_logger

__author__ = 'mchlsdrv@gmail.com'

plt.style.use('seaborn')


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

    # - Cloud utils
    s3_utils = S3Utils(
        input_bucket_configs=dict(
            name=args.aws_input_bucket_name,
            region=args.aws_input_region,
            sub_folder=args.aws_input_bucket_subdir,
        ),
        output_bucket_configs=dict(
            name=args.aws_output_bucket_name,
            region=args.aws_output_region,
            sub_folder=args.aws_output_bucket_subdir,
        ),
        delimiter='/',
        logger=logger
    )

    # - Choose device to run on
    device = get_device(gpu_id=args.gpu_id, logger=logger)

    # - Get the model
    model_configs = get_model_configs(model_name='unet', configs_dir=args.model_configs_dir, logger=logger)
    model = UNet(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        n_features=model_configs.get('n_features'),
        output_dir=args.output_dir,
        logger=logger
    ).to(device)
    load_checkpoint(model=model, checkpoint_file=args.checkpoint_file)

    # - Dir which will contain the images after detections
    output_dir = current_run_dir / 'detections'
    os.makedirs(output_dir, exist_ok=True)

    # - Augmentations for the images
    _, inf_augs = get_augs(
        args=dict(
            image_width=args.image_width,
            image_height=args.image_height,
        )
    )

    if args.infer_local:
        # - DETECT LOCAL IMAGES
        detect_images(
            image_dir=args.inference_image_dir,
            model=model,
            augs=inf_augs,
            device=device,
            save_dir=output_dir
        )
    else:
        # - Dir which will contain downloaded images from the cloud to run detection on
        input_dir = current_run_dir / 'images'
        os.makedirs(input_dir, exist_ok=True)
        while True and not args.infer_local:
            # - Downloads the images from the bucket to the local directory for the inference
            print(f'Downloading images for detection from \'{args.aws_input_bucket_name}/{args.aws_input_bucket_subdir}\' (region: {args.aws_input_region}) bucket to \'{input_dir}\'...')
            s3_utils.download_files(save_dir=input_dir, delete=DELETE_ON_FINISH)
            print(f'Images for detection were successfully downloaded from \'{args.aws_input_bucket_name}/{args.aws_input_bucket_subdir}\' (region: {args.aws_input_region}) bucket to \'{input_dir}\'!')

            # - DETECT IMAGES FROM THE CLOUD
            detect_images(
                image_dir=input_dir,
                model=model,
                augs=inf_augs,
                device=device,
                save_dir=output_dir
            )

            # - Uploads the images with detections from the local directory to the bucket
            print(f'Uploading detections from \'{output_dir}\' to \'{args.aws_output_bucket_name}/{args.aws_output_bucket_subdir}\' (region: {args.aws_output_region}) bucket...')
            s3_utils.upload_files(data_dir=output_dir, delete=DELETE_ON_FINISH)
            print(f'Detections were successfully uploaded from \'{output_dir}\' to \'{args.aws_output_bucket_name}/{args.aws_output_bucket_subdir}\' (region: {args.aws_output_region}) bucket!')
