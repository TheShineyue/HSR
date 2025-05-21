# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
import numpy as np
import random
import torch
from datasets import load_dataset, Dataset, Features, Value, Image, ClassLabel
from torch.utils.data import DataLoader,Subset
import pandas as pd
import json
from PIL import Image

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def load_vlm_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    return dataset

def get_chat_llama3(prompt, answer=None):
    if answer != None :
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
    else:
        conversation = [
        {
            "role":"user",
            "content":[
                {"type": "image"}, 
                {
                    "type":"text",
                    "text": prompt
                }
            ]
        }]
    return conversation
def get_VLguard_llama3(nsamples, seed, seqlen, tokenizer, processor, disentangle=False, data_mode=None):
    if data_mode == 'train_safes':
        data_path = '/home/liyue/psafety/dataset/VLguard/train/train_safe_safes.json'
        imgs_path = '/home/liyue/psafety/dataset/VLguard/train'
    elif data_mode == 'train_unsafes':
        data_path = '/home/liyue/psafety/dataset/VLguard/train/train_unsafes.json'
        imgs_path = '/home/liyue/psafety/dataset/VLguard/train'
    elif data_mode == 'train_safe_unsafes':
        data_path = '/home/liyue/psafety/dataset/VLguard/train/train_safe_unsafes.json'
        imgs_path = '/home/liyue/psafety/dataset/VLguard/train'
    print(data_mode)
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_vlm_dataset(data_path)
    dataset = dataset.shuffle(seed=seed).select(range(nsamples))
    trainloader = []
    torch.set_printoptions(threshold=10**6)
    for i in range(nsamples):
        image = imgs_path + '/' + dataset[i]["image"]
        prompt = dataset[i]["question"]
        answer = dataset[i]["response"]
        text_prompt = get_chat_llama3(prompt,answer)
        _text_prompt = get_chat_llama3(prompt)
        text_prompt = processor.apply_chat_template(text_prompt, add_generation_prompt=False)
        _text_prompt = processor.apply_chat_template(_text_prompt, add_generation_prompt=True)
        image = Image.open(image).convert('RGB')
        input_ids = processor(text=text_prompt, images=image, return_tensors="pt", padding=False, add_special_tokens=False)
        _input_ids = processor(text=_text_prompt, images=image, return_tensors="pt", padding=False, add_special_tokens=False)
        # print(input_ids["input_ids"])
        # print(_input_ids["input_ids"])
        labels = input_ids["input_ids"].clone()
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        _len = len(_input_ids['input_ids'][0])
        for i in range(_len - 1):
            assert _input_ids['input_ids'][0][i] == input_ids['input_ids'][0][i]
            labels[0][i] = -100
        labels[0][-1] = -100
        trainloader.append((input_ids,labels))
    return trainloader, None
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, 
    image_processor=None, tokenizer=None, processor = None, 
    data_args = None,model = None, disentangle=False,model_name = None,data_mode = None
):
    if name == 'VLguard':
        if 'llama3-llava' in model_name:
            return get_VLguard_llama3(nsamples, seed, seqlen, tokenizer, processor, disentangle = disentangle, data_mode=data_mode)
        else:
            raise ValueError("None")