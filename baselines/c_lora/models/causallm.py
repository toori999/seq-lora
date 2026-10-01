import os

import torch
import torch.nn as nn

from run import get_modelwrapper

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftConfig,
)

class CausalLM(nn.Module):
    def __init__(self, args, accelerator=None, **kwargs) -> None:
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if args.load_checkpoint:
            print('=====================')
            if args.ood_ori_dataset is not None:
                if ('c_lora' in args.modelwrapper) and (args.bayes_eval_n_samples_final == 0) : 
                    # for each seed, load the best model
                    print("load the best saved model")   
                
            else:
                if args.load_model_path is not None and os.path.exists(args.load_model_path):
                    args.load_path = args.load_model_path
                else:
                    args.load_path = f'checkpoints/{args.modelwrapper}/{args.model}/{args.dataset}/{args.load_model_path}'
            print('Loading model from: ', args.load_path)
            peft_config = PeftConfig.from_pretrained(args.load_path, is_trainable=True)
            model_kwargs = {}
            if args.load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                model_kwargs["dtype"] = torch.bfloat16
            # /teamspace/studios/this_studio/bayesian-peft/last_models/
            model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **model_kwargs)
            
            self.model = PeftModel.from_pretrained(model, args.load_path, is_trainable=True)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")
            if not args.load_in_8bit:
                self.model.to(dtype=torch.bfloat16)
            self.model.print_trainable_parameters()

            print('Model loaded successfully')
            print('=====================')
        else:
            model_kwargs = {}
            if args.load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                model_kwargs["dtype"] = torch.bfloat16
            if args.load_model_path is not None:
                model = AutoModelForCausalLM.from_pretrained(args.load_model_path, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
            if args.apply_classhead_lora:
                target_modules=["q_proj", "v_proj", "lm_head"]
            elif args.apply_qkv_head_lora:
                target_modules=["q_proj", "v_proj", "k_proj", "lm_head"]
            else:
                target_modules=["q_proj", "v_proj"]
            
            peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules)
            self.model = get_peft_model(model, peft_config)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")
            self.model.print_trainable_parameters()
