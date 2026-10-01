"""
This codebase is derived from:
    https://github.com/Wang-ML-Lab/bayesian-peft
Original License: MIT License
Copyright (c) 2025 ML@Rutgers

This repository retains much of the original structure and dependencies,
with custom modifications for our project.

-------------------------------------------------------------------------------
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-------------------------------------------------------------------------------
"""




import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
from ipdb import iex

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in [
    project_path,
    project_path + '/datasets',
    project_path + '/backbones',
    project_path + '/models',
]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
# sys.path.append(project_path + '/modelwrappers') 
main_path = project_path + '/main'
if main_path in sys.path:
    sys.path.remove(main_path)
sys.path.insert(0, main_path)

import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
from utils.args import add_management_args, add_experiment_args
from utils import create_if_not_exists
# from utils.continual_training import train as ctrain
from run.ood_eval_mdl import ood_eval
try:
    from run.laplace_train import laplace_train_old
    from run.laplace_ood_eval import laplace_ood_eval
    from run.laplace_ood_vis import laplace_ood_vis
except ModuleNotFoundError:
    laplace_train_old = None
    laplace_ood_eval = None
    laplace_ood_vis = None
from run import *

from accelerate.utils import set_seed
from accelerate import Accelerator

try:
    import wandb
except ImportError:
    wandb = None


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='Bayesian LoRA', allow_abbrev=False)
    add_management_args(parser)
    add_experiment_args(parser)
    args = parser.parse_known_args()[0]

    # add model-specific arguments
    mod = importlib.import_module('modelwrappers.' + args.modelwrapper)
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser() # the real parsing happens. 
    args = parser.parse_args()

    # set random seed
    if args.seed is not None:
        set_seed(args.seed)

    return args


def _normalize_eval_task(name):
    aliases = {
        "iid": "iid",
        "arc-c": "ARC-Challenge",
        "arcc": "ARC-Challenge",
        "arc_challenge": "ARC-Challenge",
        "ARC-C": "ARC-Challenge",
        "arc-e": "ARC-Easy",
        "arce": "ARC-Easy",
        "arc_easy": "ARC-Easy",
        "ARC-E": "ARC-Easy",
        "mmlu-chem": "MMLU-chem",
        "mmlu_chem": "MMLU-chem",
        "MMLU-chem": "MMLU-chem",
        "mmlu-phy": "MMLU-phy",
        "mmlu_phy": "MMLU-phy",
        "MMLU-phy": "MMLU-phy",
    }
    return aliases.get(name, aliases.get(name.lower(), name))


def _eval_tasks(raw):
    return [
        _normalize_eval_task(x.strip())
        for x in str(raw or "").split(",")
        if x.strip()
    ]


def _run_extra_evals(model, args, accelerator):
    tasks = _eval_tasks(getattr(args, "eval_dataset", ""))
    if not tasks:
        return

    source_dataset = args.dataset
    source_outdim = args.outdim
    source_num_samples = args.num_samples
    source_target_ids = model.model.target_ids
    source_num_classes = model.model.num_classes
    source_eval_samples = model.model.eval_n_samples
    source_testing_set = args.testing_set

    for task in tasks:
        if task == "iid":
            continue
        args.dataset = task
        args.testing_set = "auto"
        eval_dataset = get_dataset(args.dataset_type, accelerator, args)
        eval_dataset.get_loaders()
        eval_loader = accelerator.prepare(eval_dataset.test_dataloader)

        model.model.target_ids = eval_dataset.target_ids.squeeze(-1)
        model.model.num_classes = eval_dataset.num_labels
        split_name = getattr(eval_dataset, "eval_split_name", "test")
        for ns in [int(getattr(args, "bayes_eval_n_samples_final", 0) or 0)]:
            model.model.eval_n_samples = ns
            val_acc, val_ece, val_nll, val_brier = model.model.evaluate(eval_loader)
            msg = (
                f'[{source_dataset}-> {task}({split_name})][C-LoRA ns={ns}] '
                f'NLL={val_nll:.4f} ACC={val_acc*100:.2f}% '
                f'ECE={val_ece*100:.2f}% Brier={val_brier:.4f}'
            )
            print(msg)
            import logging
            logging.info(msg)

    args.dataset = source_dataset
    args.outdim = source_outdim
    args.num_samples = source_num_samples
    args.testing_set = source_testing_set
    model.model.target_ids = source_target_ids
    model.model.num_classes = source_num_classes
    model.model.eval_n_samples = source_eval_samples

# @iex
def main(args=None):
    lecun_fix()
    if args is None:
        # mclass = getattr()
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")
    
   
   
    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    accelerator = Accelerator()

    ood_ori_dataset = None
    if args.ood_ori_dataset is not None:
        dataset = args.dataset
        args.dataset = args.ood_ori_dataset
        ood_ori_dataset = get_dataset(args.dataset_type, accelerator, args)
        ood_ori_dataset.get_loaders()
        args.ood_ori_outdim = ood_ori_dataset.num_labels # should be careful to use in evaluate_all
        args.ood_ori_num_samples = ood_ori_dataset.num_samples
        args.dataset = dataset

    dataset = get_dataset(args.dataset_type, accelerator, args)
    dataset.get_loaders()
    args.outdim = dataset.num_labels 
    args.num_samples = dataset.num_samples

    model = get_model(args, accelerator)
    
    


    
    
    # set job name
    setproctitle.setproctitle('{}_{}_BLoB-lora'.format(args.model, args.dataset))

    # train the model
    if args.laplace_vis:
        if laplace_ood_vis is None:
            raise RuntimeError("Laplace visualization code is not present in this C-LoRA source checkout.")
        laplace_ood_vis(model, dataset, accelerator, args, ood_ori_dataset)
    if args.ood_ori_dataset is not None and args.laplace_train:
        if laplace_ood_eval is None:
            raise RuntimeError("Laplace OOD code is not present in this C-LoRA source checkout.")
        laplace_ood_eval(model, dataset, accelerator, args, ood_ori_dataset)
    elif args.laplace_train:
        if laplace_train_old is None:
            raise RuntimeError("Laplace training code is not present in this C-LoRA source checkout.")
        laplace_train_old(model, dataset, accelerator, args)
    
    elif args.ood_ori_dataset is not None: 
        ood_eval(model, dataset, accelerator, args, ood_ori_dataset)

    else:
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
        
        model.model.prepare_for_fit_evaluate(dataset, wandb_logger)
        model.model.fit_evaluate()
        _run_extra_evals(model, args, accelerator)
        
        
        if args.dataset == 'obqa' and (args.ood_ori_dataset is None):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_folder = f'last_models/{args.modelwrapper}/{args.model}/{args.dataset}/{args.checkpoint_dic_name}/{args.seed}'
                create_if_not_exists(save_folder)
                model.model.base_model = accelerator.unwrap_model(model.model.base_model)
                model.model.save_pretrained(save_folder, save_function = accelerator.save)


        # checkpointing the backbone model.
        if args.checkpoint: # by default the checkpoints folder is checkpoints
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # save_folder = f'checkpoints/{args.modelwrapper}/{args.model}/{args.dataset}/{args.checkpoint_dic_name}/{args.seed}'
                save_folder = f'last_models_1/{args.modelwrapper}/{args.model}/{args.dataset}/{args.checkpoint_dic_name}/{args.seed}'
                create_if_not_exists(save_folder)
                try: 
                    
                    model.model.base_model = accelerator.unwrap_model(model.model.base_model)
                    model.model.save_pretrained(save_folder, save_function=accelerator.save) ####### modified by AMIR on Feb 19 - 2024
                    
                except: 
                    print("ERROR WHILE SAVING MODEL")

        
       

        if not args.nowand:
            if accelerator.is_local_main_process:
                wandb_logger.finish()


if __name__ == '__main__':
    main()
