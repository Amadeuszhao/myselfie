import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer_id = "/root/autodl-tmp/base/model"
tokenizer =  AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token# %%
def generate_prompt(data_point):
    """
    Generate input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenized prompt
    """
    if "[INST]" not in  data_point["text"]:
        text = '[INST] ' + data_point["text"]
    else:
        text =  data_point["text"]

    return text

# %%
from datasets import load_dataset
# data = load_dataset("json",data_files="/root/autodl-tmp/finetune/ft_datasets_mistral_addharmful.json", split='train[:100%]')

data = load_dataset("json",data_files="/root/autodl-tmp/finetune/llama2_train_alpaca.json", split='train[:100%]')
df = data.to_pandas()
    
df

text_column = [generate_prompt(data_point) for data_point in data]

data = data.add_column("prompt", text_column)

data = data.shuffle(seed=1234)
data = data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

data = data.train_test_split(test_size=0.0001)
train_data = data["train"]
print("This is size of train data", len(train_data))
test_data = data["test"]

import torch
import datasets
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    TrainerCallback,
    Trainer,
)
import transformers
import random
from lat_trainer import LATSFTTrainer, EvaluateFirstStepCallback
from lat_model import LATLlamaForCausalLM
import argparse
from datetime import datetime
import pickle
from tqdm import tqdm

model_id = "/root/autodl-tmp/base/model"
model = LATLlamaForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16,
    device_map='balanced_low_0'
            )

args=transformers.TrainingArguments(
            per_device_train_batch_size=1,        
            gradient_accumulation_steps=4,        
            warmup_steps=int(0.03 * 12000),                    
            max_steps=12000,                        
            learning_rate=5e-5,                   
            fp16=False,                            
            logging_steps=1,                      
            output_dir=f"adver_train_model_layer16_1618_epsilon_12_steps_4_12000",  
            optim="paged_adamw_8bit",             
            save_strategy="steps", 
            save_steps=4000,
            push_to_hub=False 
)
# %%
def print_trainable_parameters(model):
    """
        Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
        
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        
    print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

# Assuming 'original_model' is your instance of LlamaForCausalLM
freeze_parameters(model)

# Enable gradient for newly added layers
#for param in model.extra_layer_1.parameters():
#    param.requires_grad = True
#for param in model.extra_layer_1.parameters():
#    param.requires_grad = True
for i in range(16, 18):
    for param in model.model.layers[i].parameters():
        param.requires_grad = True
for param in model.lm_head.parameters():
    param.requires_grad = False
print_trainable_parameters(model)

epsilon = 12
perturb_layer = 16
steps = 4
norm_type = 'l2'

trainer = LATSFTTrainer(
        model=model,
        train_dataset=train_data,  
        max_seq_length=512,  # None
        eval_dataset=test_data,
        dataset_text_field='text',
        tokenizer=tokenizer,
        args=args,
        perturb_layer=perturb_layer,
        epsilon=epsilon,
        steps=steps,
        norm_type= norm_type,
        random_init=True,
        std_normalization=False,
        keep_in_eval=True,
        perturb_target='residuals',
    )
trainer.train()