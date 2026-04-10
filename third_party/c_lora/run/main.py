import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
from pathlib import Path

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
sys.path.append(project_path + "/datasets")
sys.path.append(project_path + "/models")
sys.path.append(project_path + "/main")

import datetime
import uuid
from argparse import ArgumentParser

try:
    import setproctitle
except Exception:
    setproctitle = None

import torch
from utils.args import add_management_args, add_experiment_args
from utils import create_if_not_exists, StageTimer
from run import *

from accelerate.utils import set_seed
from accelerate import Accelerator

try:
    import wandb
except ImportError:
    wandb = None


def lecun_fix():
    from six.moves import urllib  # pyright: ignore

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description="C-LoRA", allow_abbrev=False)
    add_management_args(parser)
    add_experiment_args(parser)
    args = parser.parse_known_args()[0]

    mod = importlib.import_module("modelwrappers." + args.modelwrapper)
    get_parser = getattr(mod, "get_parser")
    parser = get_parser()
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    return args


def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    accelerator = Accelerator()
    method_tag = str(args.modelwrapper).upper()

    with StageTimer(f"LOAD-STAGE {method_tag} on {args.dataset}"):
        dataset = get_dataset(args.dataset_type, accelerator, args)
        dataset.get_loaders()
        args.outdim = dataset.num_labels
        args.num_samples = dataset.num_samples

        if setproctitle is not None:
            setproctitle.setproctitle(f"{args.model}_{args.dataset}_{method_tag}")

        wandb_logger = None
        if accelerator.is_local_main_process:
            print(args)
            if not args.nowand:
                assert wandb is not None, "Wandb not installed, please install it or run without wandb"
                if not args.wandb_name:
                    wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
                else:
                    wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))
            print(file=sys.stderr)

        model = get_model(args, accelerator)
        modelwrapper = get_modelwrapper(args.modelwrapper)
        model.model = modelwrapper(
            model.model, model.peft_config, args, accelerator, adapter_name="default"
        )
        model.model.print_trainable_parameters()
        model.model.prepare_for_fit_evaluate(dataset, wandb_logger)

    model.model.fit_evaluate()

    if args.checkpoint:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_folder = f"checkpoints/{args.modelwrapper}/{args.model}/{args.dataset}/{args.checkpoint_dic_name}/{args.seed}"
            create_if_not_exists(save_folder)
            model.model.base_model = accelerator.unwrap_model(model.model.base_model)
            model.model.save_pretrained(save_folder, save_function=accelerator.save)
            print("Model saved to:", save_folder)

    if not args.nowand and accelerator.is_local_main_process and wandb_logger is not None:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
