# %% [markdown]
# # Latent Space Adversarial Attack
# 
# This notebook demonstrates conducting a latent space adversarial attacks on LLMs. These particualr demo attacks are created using projected gradient descent to make a model more jailbreakable. 
# 
# ## Imports

# %%
import time
import json
import os
import torch
import sys
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
os.chdir("../")
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from latent_at import *
from tasks.harmbench.HarmBenchTask import HarmBenchTask

load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

# %%
## Configuration

# Set whether to use Llama2-7B or Llama3-8B.

# %%
use_llama2 = True
if use_llama2:  # use llama2-7b
    model_name = "/root/autodl-tmp/base/model"
else: # use llama3-8b
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# %% [markdown]
# ## Model

# %%
model_dtype = torch.bfloat16
device = "cuda"
run_start_evals = False

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_access_token,
    torch_dtype=model_dtype
).to(device)


if "Llama-2" or 'base' in model_name:
    model_type = "llama2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
elif "Llama-3" in model_name:
    model_type = "llama3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
elif "zephyr" in model_name or "mistral" in model_name:
    model_type = "zephyr"    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"
else:
    print(model_name)
    raise Exception("Unsupported model type?")

# %% [markdown]
# ## Data

# %%
advbench_data = HarmBenchTask(
    tokenizer=tokenizer,
    gen_batch_size=1,
    cls_batch_size=1,
    device=device,
    data_name="advbench",
    train_test_split=.8
)

harmbench_data = HarmBenchTask(
    tokenizer=tokenizer,
    gen_batch_size=1,
    cls_batch_size=1,
    device=device,
    data_name="harmbench_text",
    train_test_split=.8,
    func_categories=["standard", "contextual"]
)

# train_behaviors = advbench_data.train_behaviors + harmbench_data.train_behaviors  

sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

if model_type == "llama2":
    use_tokenizer_template = False
    custom_prompt_template = "[INST] {prompt} [/INST] \n"
    custom_completion_template = "{completion}"
elif model_type == "llama3":
    use_tokenizer_template = False
    custom_prompt_template = "<|start_header_id|>user<|end_header_id|\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    custom_completion_template="{completion}"
else:
    sys_prompt=""
    use_tokenizer_template = False
    custom_prompt_template = "<|user|>\n{prompt}</s> \n <|assistant|>\n"
    custom_completion_template="{completion}"
 
lat_dataset = process_generic_chat_dataset(
    tokenizer,
    dataset="LLM-LAT/harmful-dataset",
    adv_column="rejected",
    def_column="chosen",
    split="train",
    use_tokenizer_template=use_tokenizer_template,
    system_prompt="",
    custom_prompt_template=custom_prompt_template,
    custom_completion_template=custom_completion_template
)

lat_dataloader = DataLoader(
    lat_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True,
    collate_fn=LatentAdversarialTrainingDataCollator(
        tokenizer.pad_token_id,
        truncate_length=2048
    )
)

dataloader = iter(lat_dataloader)

# %%
batch = next(dataloader)
import os
import shutil

# 文件夹路径
folder_path = '/root/autodl-tmp/at/uap'

# 检测文件夹是否存在
if os.path.exists(folder_path):
    # 如果文件夹存在，删除文件夹
    shutil.rmtree(folder_path)

# 创建新的文件夹
os.makedirs(folder_path)

print(f'文件夹 {folder_path} 已创建。')
# %%
tokenizer.decode(batch['adv_padd_prompt_tokens'][0][40-20:40])

# %%
adv_padd = batch['adv_padd_prompt_tokens']
def_padd = batch['def_padd_prompt_tokens']
prompt_mask = batch['padd_prompt_mask']
def_padd_mask = batch['adv_padd_labels_mask']
adv_padd_mask = batch['adv_padd_labels_mask']
print(tokenizer.decode(adv_padd[adv_padd_mask]))
tokenizer.decode(def_padd[def_padd_mask])
embedding_space = np.load("/root/autodl-tmp/myselfie/embeddings.npy")
# %%
data = []


# %% [markdown]
# ## Clean Performance

# %% [markdown]
# ## Latent Space Attack
start_time = time.time()

# %%
losses, wrappers = projected_gradient_descent_onpadd(
    batch=batch,  # batch
    model=model,  # model
    model_layers_module="model.layers",  # where the model layers are
    layer=["embedding", ],  # layers to attack
    epsilon=0.5496,  # attack l2 constraint
    l2_regularization=0.0,  # coef for l2 penalty on the attack
    learning_rate=5e-2,  # attack step size
    pgd_iterations=1000,  # how many steps of projected gradient descent to do
    loss_coefs={"toward": 0.8, "away": 0.5,},  # coefficients for the attack's toward and away losses
    log_loss=True,  # whether to use a log loss instead of a crossentropy one
    return_loss_over_time=True,
    device="cuda",
    embedding_constraint=False,
    embedding_method = 'model',
    embedding_space=embedding_space
)
end_time = time.time()
execution_time = end_time - start_time
print(f"代码执行时间为: {execution_time} 秒")
print("***ADVERSARIAL LOSSES OVER TIME***\n")
print([round(l['adv_total'], 4) for l in losses])

# %% [markdown]
# ## Attacked Performance

# %%
for wrapper in wrappers:
    wrapper.enabled = True  # the wrappers should already be enabled, so this is redundant
with torch.no_grad():
    outputs = model.generate(
        batch["adv_padd_prompt_tokens"][0][0:batch["prompt_length"][0][2]].unsqueeze(0).to("cuda"),
        max_length=batch["adv_padd_prompt_tokens"].shape[1] + 200,
    )

print("***ATTACKED PERFORMANCE***\n")
prompt = tokenizer.decode( batch["adv_padd_prompt_tokens"][0][0:batch["prompt_length"][0][2]]).replace('\n', '')
print("Prompt:\n" + prompt + "\n")
prompt_response = tokenizer.decode(outputs[0]).replace('\n', '')
print("Completion:\n" + prompt_response[len(prompt):])
data.append({
            'losses': [round(l['adv_total'], 4) for l in losses],
            'prompt': prompt,
            'prompt_response': prompt_response[len(prompt):]
        })
with open('/root/autodl-tmp/at/output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

for batch in dataloader:
    losses, wrappers = projected_gradient_descent_onpadd(
    batch=batch,  # batch
    model=model,  # model
    model_layers_module="model.layers",  # where the model layers are
    layer=["embedding", ],  # layers to attack
    epsilon=0.5496,  # attack l2 constraint
    l2_regularization=0.0,  # coef for l2 penalty on the attack
    learning_rate=5e-2,  # attack step size
    pgd_iterations=1000,  # how many steps of projected gradient descent to do
    loss_coefs={"toward": 0.8, "away": 0.5,},  # coefficients for the attack's toward and away losses
    log_loss=True,  # whether to use a log loss instead of a crossentropy one
    return_loss_over_time=True,
    device="cuda",
    embedding_constraint=False,
    embedding_method = 'model',
    embedding_space=embedding_space
)

    print("***ADVERSARIAL LOSSES OVER TIME***\n")
    print([round(l['adv_total'], 4) for l in losses])
    for wrapper in wrappers:
        wrapper.enabled = True  # the wrappers should already be enabled, so this is redundant
    with torch.no_grad():
        outputs = model.generate(
            batch["adv_padd_prompt_tokens"][0][0:batch["prompt_length"][0][2]].unsqueeze(0).to("cuda"),
            max_length=batch["adv_padd_prompt_tokens"].shape[1] + 200,
        )

    print("***ATTACKED PERFORMANCE***\n")
    prompt = tokenizer.decode( batch["adv_padd_prompt_tokens"][0][0:batch["prompt_length"][0][2]]).replace('\n', '')
    print("Prompt:\n" + prompt + "\n")
    prompt_response = tokenizer.decode(outputs[0]).replace('\n', '')
    print("Completion:\n" + prompt_response[len(prompt):])
    data.append({
            'losses': [round(l['adv_total'], 4) for l in losses],
            'prompt': prompt,
            'prompt_response': prompt_response[len(prompt):]
        })
    with open('/root/autodl-tmp/at/output.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



