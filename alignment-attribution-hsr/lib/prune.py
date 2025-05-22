import os
import time
import heapq
import torch
import torch.nn as nn
import pickle
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT, WrappedGPTJMLLM
from .data import get_loaders
import json
import random
from .ablate import AblateGPT
import heapq
import transformers
import numpy as np
import math

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(
        sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    # print(module.named_children())
    for name1, child in module.named_children():
        # print(name1)
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res

def check_sparsity(args,model):
    layers = model.language_model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
    # model.config.use_cache = use_cache
    return float(count), float(count) / total_params

def prepare_calibration_input(model, dataloader, device, nsamples):
    layers = model.language_model.model.layers
    dtype = next(iter(model.parameters())).dtype
    print(dtype)
    inps = []
    tars = []
    attention_mask = []
    position_ids = []
    # torch.set_printoptions(threshold=10**6)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, input_ids, **kwargs):
            # print(kwargs)
            # assert 1==0
            inps.append(input_ids)
            attention_mask.append(kwargs["attention_mask"])
            position_ids.append(kwargs["position_ids"])
            raise ValueError

    layers[0] = Catcher(layers[0])
    cnt = 0
    for batch in dataloader:
        try:
            # print(cnt)
            cnt+=1
            torch.cuda.empty_cache()
            tars.append(batch[1])
            model(
                input_ids = batch[0]["input_ids"].to(device),
                pixel_values = batch[0]["pixel_values"].to(device),
                image_sizes = batch[0]["image_sizes"].to(device),
            )
            torch.cuda.empty_cache()
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = [None for _ in range(nsamples)]
    # assert 1==0
    return inps, outs, tars, attention_mask, position_ids

def prune_lvlm_wanda_hf(
    args,
    model,
    tokenizer,
    processor,
    model_base=None,
    data_mode = None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="to do",
):
    print(f"loading calibration data {prune_data}")
    dataloader, _ = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        processor=processor,
        disentangle=args.disentangle,
        model_name = args.model,
        data_mode = data_mode
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's

    inps = [inp.squeeze(0).to(device) for inp in inps]
    tars = [tar.squeeze(0).to(device) for tar in tars]
    position_ids = [pids.to(device) for pids in position_ids]
    prune_data = prune_data +'_' + data_mode

    layers = model.language_model.model.layers

    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)
            return tmp
        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name, tars[j]))
                )
            with torch.inference_mode():
                outs[j] = layer(inps[j].unsqueeze(0), 
                            attention_mask=attention_mask[j],
                            position_ids=position_ids[j])[0]
            for h in handles:
                h.remove()
        # prune_data = prune_data+'_' + data_mode
        if not args.prune_part:
            for name in subset:
                # print()
                print(f"pruning layer {i} name {name}")
                if args.use_diff or args.recover_from_base:
                    magnitude = torch.abs(
                        subset[name].weight.data - subset_base[name].weight.data
                    )
                else:
                    magnitude = torch.abs(subset[name].weight.data)
                act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = magnitude * act

                if args.neg_prune:
                    W_metric = -W_metric

                if args.dump_wanda_score:
                    # Only save the score, no pruning
                    if args.use_diff:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"{args.sparsity_ratio}/wanda_score/{prune_data}_weight_diff_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save, f"{args.sparsity_ratio}/wanda_score/{prune_data}_weight_diff"
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff.pkl",
                            )
                            
                    else:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"{args.sparsity_ratio}/wanda_score/{prune_data}_weight_only_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save, f"{args.sparsity_ratio}/wanda_score/{prune_data}_weight_only"
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only.pkl",
                            )
                    with open(target_file, "wb") as f:
                        print(
                            "Writing W_metric in layer {} and name {} with {} to the file".format(
                                i, name, prune_data
                            )
                        )
                        pickle.dump(W_metric, f)
                    continue

                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii : (ii + prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0.0, 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                        while (
                            torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][
                            :, : int(W_metric.shape[1] * args.sparsity_ratio)
                        ]
                        W_mask.scatter_(1, indices, True)

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
                # del W_mask
                # torch.cuda.empty_cache()
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = (
                    ((i == 0) and (name == "mlp.down_proj"))
                    or (
                        (i == 1)
                        and ((name == "self_attn.o_proj") or (name == "mlp.down_proj"))
                    )
                    or (
                        (i > 1)
                        and (
                            (name == "self_attn.o_proj")
                            or (name == "mlp.gate_proj")
                            or (name == "mlp.down_proj")
                            or (name == "mlp.up_proj")
                        )
                    )
                )
                if condition:
                    print(f"pruning layer {i} name {name}")
                    if args.use_diff or args.recover_from_base:
                        magnitude = torch.abs(
                            subset[name].weight.data - subset_base[name].weight.data
                        )
                    else:
                        magnitude = torch.abs(subset[name].weight.data)
                    act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    W_metric = magnitude * act

                    del act

                    if args.neg_prune:
                        W_metric = -W_metric

                    if args.dump_wanda_score:
                        # Only save the score, no pruning
                        if args.use_diff:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"{args.sparsity_ratio}/wanda_score/{prune_data}_weight_diff_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save, f"{args.sparsity_ratio}/wanda_score/{prune_data}_weight_diff"
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff.pkl",
                                )
                        else:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"{args.sparsity_ratio}/wanda_score/{prune_data}_weight_only_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save, f"{args.sparsity_ratio}/wanda_score/{prune_data}_weight_only"
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{prune_data}_weight_only.pkl",
                                )
                        with open(target_file, "wb") as f:
                            print(
                                "Writing W_metric in layer {} and name {} with {} to the file".format(
                                    i, name, prune_data
                                )
                            )
                            pickle.dump(W_metric, f)
                        continue

                    W_mask = (
                        torch.zeros_like(W_metric) == 1
                    )  ## initialize a mask to be all False
                    if prune_n != 0:
                        # structured n:m sparsity
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii : (ii + prune_m)].float()
                                W_mask.scatter_(
                                    1,
                                    ii
                                    + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                    True,
                                )
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)

                        if args.use_variant:
                            # wanda variant
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)

                            alpha = 0.4
                            alpha_hist = [0.0, 0.8]
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                            while (
                                torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                            ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > args.sparsity_ratio:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha

                                alpha = alpha_new
                                W_mask, cur_sparsity = return_given_alpha(
                                    alpha, sort_res, W_metric, tmp_metric, sum_before
                                )
                            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            # unstructured pruning
                            indices = sort_res[1][
                                :, : int(W_metric.shape[1] * args.sparsity_ratio)
                            ]
                            W_mask.scatter_(1, indices, True)

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[
                            name
                        ].weight.data[
                            W_mask
                        ]  # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero
                    del W_metric
                    torch.cuda.empty_cache()

        for j in range(args.nsamples):
            with torch.inference_mode():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0].squeeze(0)

        inps, outs = outs, inps
    del subset, wrapped_layers
    torch.cuda.empty_cache()


def prune_lvlm_wanda_recover_heads(
    args,
    model,
    tokenizer,
    processor,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_safe_data="VLguard_train_unsafes",
    prune_use_data="VLguard_train_safes",
    p=0.5,
    q=0.5,
    max_p=0.7,
    only_heads=True, # add
    top_h=7,
    heads_paths = "/home/liyue/psafety/SafetyHeadAttribution-main/exp_res"
):
    top_h = int(top_h)
    with open(heads_paths,'r',encoding='utf-8') as fhead:
        data = json.load(fhead)
    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)
    heads_scores = sorted_data[:top_h]
    layers = model.language_model.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers
    metric1 = prune_use_data
    metric2 = prune_safe_data
    print(
        "prune p = {}, q = {}, with metric1 = {}, metric2 = {}".format(
            p, q, metric1, metric2
        )
    )
    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])
        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.model == "llava-next-vicuna-hf-7b":
                    W_metric1 = pickle.load(
                        open(
                            f"/home/liyue/psafety/LVLM_prune_safety/out/llava-next-vicuna-hf-7b/unstructured/lvlm_wanda_hf_weightonly/{args.sparsity_ratio}/wanda_score/{metric1}_weight_only_disentangle/W_metric_layer_{i}_name_{name}_{metric1}_weight_only_disentangle.pkl",
                            "rb",
                        )
                    ).to(device)
                    W_metric2 = pickle.load(
                        open(
                            f"/home/liyue/psafety/LVLM_prune_safety/out/llava-next-vicuna-hf-7b/unstructured/lvlm_wanda_hf_weightonly/{args.sparsity_ratio}/wanda_score/{metric2}_weight_only_disentangle/W_metric_layer_{i}_name_{name}_{metric2}_weight_only_disentangle.pkl",
                            "rb",
                        )
                    ).to(device)
                else:
                    raise NotImplementedError
                top_p = int(p * W_metric1.shape[1] * W_metric1.shape[0])  # top_p utility
                top_max_p = int(max_p* W_metric1.shape[1] * W_metric1.shape[0])
                top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])  # top_q safety

                top_p_indices = torch.topk(W_metric1.flatten(), top_p, largest=True)[1]
                top_max_p_indices = torch.topk(W_metric1.flatten(), top_max_p, largest=True)[1]
                top_q_indices = torch.topk(W_metric2.flatten(), top_q, largest=True)[1]

                unique_p = torch.unique(top_p_indices)
                unique_max_p = torch.unique(top_max_p_indices)
                unique_q = torch.unique(top_q_indices)
                mask = (torch.isin(unique_q, unique_max_p)) & (~torch.isin(unique_q, unique_p))

                filtered_indices = unique_q[mask]

                weight_dim = subset[name].weight.data.shape[1]
                filtered_indices_rows = filtered_indices // weight_dim
                filtered_indices_cols = filtered_indices % weight_dim

                W_mask = (torch.zeros_like(W_metric1) == 1)
                sort_res = torch.sort(W_metric1, dim=-1, stable=True)
                indices = sort_res[1][:, : int(W_metric1.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

                if only_heads:
                    pass
                else:
                    if  'self_attn' in name:
                        head_nums = 32
                        for head in heads_scores:# TodO
                            layer_id, head_id = head
                            layer_id = int(head[0].split('-')[0])
                            head_id = int(head[0].split('-')[1])
                            if layer_id != i:
                                continue
                            head_dim = W_metric1.shape[1] // head_nums
                            col_start = head_id * head_dim
                            col_end = (head_id + 1) * head_dim
                            if name in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]:
                                for row, col in zip(filtered_indices_rows, filtered_indices_cols):
                                    if col_start <= col and col < col_end:  
                                        W_mask[row, col] = False
                            elif name == "self_attn.o_proj":
                                for row, col in zip(filtered_indices_rows, filtered_indices_cols):
                                    W_mask[row, col] = False     # remove
                                # 
                            
                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
