from datasets import load_dataset, Dataset, Image
from lib.utils.get_model import get_model, get_tokenizer
from lib.Sahara.svd import compute_subspace_similarity, compute_subspace_spectral_norm
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import torch
import os
import json
import random
from PIL import Image

default_search_cfg = {
    "search_step": 1,
    "mask_qkv": ['q', 'k', 'v'],
    "scale_factor": 1e-5,
    "mask_type": "scale_mask"
}

def load_from_search_cfg(search_cfg):
    temp = deepcopy(search_cfg)
    _ = temp.pop('search_step')
    return temp

def update_mask_cfg(mask_cfg, layer, head, temp=True):
    now_mask_key = (layer, head)
    new_mask_cfg = deepcopy(mask_cfg)
    if 'head_mask' not in new_mask_cfg:
        new_mask_cfg['head_mask'] = {}
    # if temp == True:
    new_mask_cfg['head_mask'][now_mask_key] = mask_cfg['mask_qkv']
    return new_mask_cfg


def get_last_hidden_states(model, tokenizer, data, device = 'cpu', mask_cfg=None):
    with torch.no_grad():
        last_hidden_states = []
        head_mask = mask_cfg['head_mask'] if mask_cfg is not None else None
        mask_type = mask_cfg['mask_type'] if mask_cfg is not None else None
        scale_factor = mask_cfg['scale_factor'] if mask_cfg is not None else None

        for batch in data:
            # batch = batch.to(device)
            # print(batch)
            input_ids = batch['input_ids']
            pixel_values = batch['pixel_values']
            image_sizes = batch['image_sizes']
            attention_mask = batch['attention_mask']
            # print(input_ids.shape)
            outputs = model(input_ids = input_ids.to(device),
                            pixel_values = pixel_values.to(device),
                            image_sizes = image_sizes.to(device),
                            attention_mask = attention_mask.to(device),
                            # position_ids = batch['position_ids'],
                            head_mask=head_mask,
                            mask_type=mask_type,
                            scale_factor=scale_factor,
                            output_hidden_states=True,
                           )
            now_lhs = outputs.hidden_states[-1]
            last_hidden_states.append(now_lhs[:, -1, :].reshape(-1))
        last_hidden_states = torch.stack(last_hidden_states)
        return last_hidden_states


def get_safety_subspace_shifts(base_last_hidden_states, last_hidden_states):
    shifts = compute_subspace_similarity(base_last_hidden_states, last_hidden_states)
    return shifts


def get_most_important_subspace(shifts_dict):
    sorted_dict = sorted(shifts_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_dict[0][0][0], sorted_dict[0][0][1]

def load_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    return dataset

def get_chat(prompt):
    conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
            },
            {
                "type":"text",
                "text": prompt
            }
        ]
    }]
    return conversation

def get_dataset(processor, data_path, imgs_path, seed, nsamples):
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_dataset(data_path)
    dataset = dataset.shuffle(seed=seed).select(range(nsamples))

    trainloader = []
    for i in range(nsamples):
        image = imgs_path + '/' + dataset[i]["image"]
        prompt = dataset[i]["question"]
        conversation = get_chat(prompt)
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image = Image.open(image).convert('RGB')
        input_ids = processor(text=text_prompt, images=image, return_tensors="pt", padding=True)
        trainloader.append(input_ids)
    return trainloader

def safety_head_attribution(model_name, data_path, imgs_path, storage_path=None, search_cfg=None, device='cuda:0', seed = 114514, nsamples = 128):
    processor, _as = get_tokenizer(model_name)
    tokenizer = processor.tokenizer
    model, _acc = get_model(model_name, get_custom=True, add_size=False)
    model = model.to(device)

    data = get_dataset(processor, data_path, imgs_path, seed, nsamples)

    layer_nums = model.language_model.config.num_hidden_layers
    head_nums = model.language_model.config.num_attention_heads

    print(layer_nums,head_nums)

    search_step = search_cfg.get('search_step', 1)
    mask_cfg = load_from_search_cfg(search_cfg)
    base_lhs = get_last_hidden_states(model, tokenizer, data, device)
    if storage_path is not None:
        all_lhs = {}
    else:
        all_lhs = None
    for step in range(search_step):
        shifts_dict = {}
        for layer in tqdm(range(0, layer_nums)):
            for head in range(0, head_nums):
                with torch.no_grad():
                    now_mask_cfg = update_mask_cfg(mask_cfg, layer, head)
                    if 'head_mask' in mask_cfg and (layer, head) in mask_cfg['head_mask']:
                        continue
                    last_hs = get_last_hidden_states(model, tokenizer, data, device = device, mask_cfg=now_mask_cfg)
                    if all_lhs is not None:
                        all_lhs[(layer, head)] = last_hs.detach().cpu()
                    # shifts = get_safety_subspace_shifts(base_lhs, last_hs)
                    shifts = compute_subspace_spectral_norm(base_lhs, last_hs)
                    print(f"{layer}, {head}"
                          f", {shifts}"
                          f"")
                    shifts_dict[(layer, head)] = shifts
        if storage_path is not None:
            torch.save(all_lhs, f"{storage_path}/{data_path.split(sep='/')[-1]}_{step}.pt")
            with open(f"{storage_path}/{data_path.split(sep='/')[-1]}_{step}.jsonl", "w+") as shifts_file:
                new_shifts_dict = {f"{k[0]}-{k[1]}": v for k, v in shifts_dict.items()}
                shifts_file.write(json.dumps(new_shifts_dict) + "\n")
        best_layer, best_head = get_most_important_subspace(shifts_dict)
        print(best_layer, best_head)
        mask_cfg = update_mask_cfg(mask_cfg, best_layer, best_head, temp=False)


if __name__ == '__main__':

    model_path = "/home/liyue/model/llama3-llava-next-8b-hf"
    data_path = "/home/liyue/psafety/dataset/VLguard/train/train_unsafes.json"
    imgs_path = '/home/liyue/psafety/dataset/VLguard/train'
    storage_path = "/home/liyue/psafety/SafetyHeadAttribution-hsr/exp_res/llama3-llava/"


    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    safety_head_attribution(
        model_name=model_path,
        data_path=data_path,
        imgs_path=imgs_path,
        search_cfg=default_search_cfg,
        storage_path=storage_path,
        device=torch.device("cuda:6")
    )