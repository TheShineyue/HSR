import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import numpy as np
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
torch.backends.cudnn.enabled = False
from lib.prune import (
    check_sparsity,
    prune_lvlm_wanda_hf,
    prune_lvlm_wanda_recover_heads,
)
from lib.model_wrapper_low import make_low_rank
from transformers import LlavaNextProcessor,LlavaNextForConditionalGeneration

print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus: ", torch.cuda.device_count())

SAVE_PATH = "temp"

modeltype2path = {
    "llava-next-vicuna-hf-7b" : "/home/liyue/model/llava-v1.6-vicuna-7b-hf",
}

def get_model(model_name, cache_dir="./model_weights", device= None):
    if model_name in [
        "llama2-7b-chat-hf",
        "llama2-13b-chat-hf",
        "llama2-7b-hf",
        "llama2-13b-hf",
    ]:
        model = AutoModelForCausalLM.from_pretrained(
            modeltype2path[model_name],
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map={"model": "cuda:6", "lm_head": "cuda:6"},
        )
        model.seqlen = model.config.max_position_embeddings
        return model
    elif model_name in ['llava-next-vicuna-hf-7b']: 
        model = LlavaNextForConditionalGeneration.from_pretrained(
            modeltype2path[model_name],
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            cache_dir=cache_dir,
            # device_map = "auto" # balanced_low_0
        ).to(device)
        model.seqlen = model.config.text_config.max_position_embeddings
        return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument("--model_base", type=str, default="llama2-7b-hf")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples."
    )
    parser.add_argument(
        "--heads_random_seed", type=int, default=-1
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0, help="Sparsity level"
    )
    parser.add_argument(
        "--sparsity_type",
        type=str,
        choices=["unstructured", "4:8", "2:4"],
        default="unstructured",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "lvlm_wanda_hf",
            "lvlm_wanda_recover_heads_gqa",
        ],
    )
    parser.add_argument(
        "--prune_data",
        type=str,
        choices=[
            "VLguard",
        ],
    )
    parser.add_argument("--data_mode",default="train_safes")
    parser.add_argument("--use_diff", action="store_true") # 使用则为true
    parser.add_argument("--neg_prune", action="store_true")
    parser.add_argument("--recover_from_base", action="store_true")
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Use combined with wandg_set_difference, the top p scored elements in the first set (alpaca_no_safety)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.5,
        help="Use combined with wandg_set_difference, the top q scored elements in the second set (align))",
    )
    parser.add_argument(
        "--max_p",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--max_q",
        type=float,
        default=0.75
    )

    parser.add_argument("--cache_dir", default="model_weights", type=str)
    parser.add_argument(
        "--use_variant",
        action="store_true",
        help="whether to use the wanda variant described in the appendix",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save results.")
    parser.add_argument(
        "--save_model", type=str, default=None, help="Path to save the pruned model."
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=None,
        help="Path to save the pruned model weight mask.",
    )
    parser.add_argument(
        "--dump_wanda_score", action="store_true", help="Whether to dump wanda scores."
    )

    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--eval_attack", action="store_true")
    parser.add_argument("--save_attack_res", action="store_true")
    parser.add_argument(
        "--prune_part", # false
        action="store_true",
        help="whether to only prune the layer with lower jaccard index",
    )
    parser.add_argument(
        "--entangle_prompt_feat",
        dest="disentangle",
        action="store_false",
        help="entangle the prompt and response when computing the wanda score",
    )
    parser.add_argument(
        "--decouple_align_utility",
        action="store_true",
        help="whether to decouple the align and utility when computing the wanda score",
    )
    parser.add_argument(
        "--decouple_align_misalign",
        action="store_true",
        help="whether to decouple the align and misalign when computing the wanda score",
    )
    parser.add_argument(
        "--only_heads",
        action="store_true"
    )
    parser.add_argument(
        "--top_h",
        default=3
    )
    parser.add_argument(
        "--op",
        default=1
    )
    parser.add_argument(
        "--heads_paths",
        default=None
    )
    parser.add_argument(
        "--device",
        default="cuda:7"
    )
    # low rank
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--niter", type=int, default=20)

    args = parser.parse_args()

    print("Disentangle:", args.disentangle) # 默认true

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    device = torch.device(args.device)
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert (
            args.sparsity_ratio == 0.5
        ), "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    print(f"loading model {args.model}")
    model = get_model(args.model, args.cache_dir, args.device)

    if args.model in ["llama3-llava-next-8b-hf"]:
        model.eval()
        processor = LlavaNextProcessor.from_pretrained(modeltype2path[args.model])
        tokenizer = processor.tokenizer

    if (args.decouple_align_misalign or args.decouple_align_utility) and (
        tokenizer.pad_token is None
    ):
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    if args.use_diff or args.recover_from_base:
        print(f"loading llm base model {args.model_base}")
        model_base = get_model(args.model_base, args.cache_dir)
        model_base.eval()
    else:
        model_base = None

    if args.decouple_align_utility or args.decouple_align_misalign:
        if args.decouple_align_utility:
            print(f"decoupling align and utility, loading extra model{args.model}")
        else:
            print(f"decoupling align and misalign, loading extra model{args.model}")
        model_extra = get_model(args.model, args.cache_dir)
        model_extra.eval()
        model_extra.resize_token_embeddings(len(tokenizer))
    else:
        model_extra = None

    
    if (
        "30b" in args.model or "65b" in args.model
    ):  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "lvlm_wanda_recover_heads":
            prune_lvlm_wanda_recover_heads(
                args,
                model,
                tokenizer,
                processor,
                model_base,
                device = device,
                prune_n = prune_n,
                prune_m = prune_m,
                prune_safe_data="VLguard_train_unsafes",
                prune_use_data="VLguard_train_safes",
                p=args.p,
                q=args.q,
                only_heads = args.only_heads,
                top_h = args.top_h,
                heads_paths = args.heads_paths
            )
        elif args.prune_method == "lvlm_wanda_hf":
            prune_lvlm_wanda_hf(
                    args,
                    model,
                    tokenizer,
                    processor,
                    model_base,
                    data_mode=args.data_mode,
                    device = device,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    prune_data=args.prune_data,
                )     

    if args.prune_method == "low_rank":
        make_low_rank(args, model, tokenizer, device, prune_data=args.prune_data)

    del model_base
    del model_extra

    ################################################################
    print("*" * 30)
    if not args.recover_from_base and args.sparsity_ratio > 0:
        count, sparsity_ratio = check_sparsity(args, model)
    else:
        sparsity_ratio = args.sparsity_ratio
    print(f"sparsity sanity check {sparsity_ratio:.8f} with {count:.8f}")
    print("*" * 30)
    ################################################################
    
    if not args.dump_wanda_score:
        print("save start...")
        if args.prune_data == 'VLguard':
            add_path = '_' + args.data_mode
        else:
            add_path = ''
        if args.sparsity_ratio == 0.5:
            final_path = '0_5'
        elif args.sparsity_ratio == 0.6:
            final_path = '0_6'
        elif args.sparsity_ratio == 0.4:
            final_path = '0_4'

        lvlm_prune_path = os.path.join(
                SAVE_PATH,
                f"{args.model}_{args.prune_data}{add_path}_{args.prune_method}_{final_path}"
            )
        # lvlm_magnitude_hf
        if args.sparsity_type != "unstructured":
            lvlm_prune_path = os.path.join(
                lvlm_prune_path,
                f"_{args.sparsity_type}"
            )
        if "lvlm_wanda_recover_heads" in args.prune_method:
            flag = "T" if args.only_heads else "F" 
            print(flag, args.top_h)
            lvlm_prune_path = os.path.join(
                lvlm_prune_path,
                f"_{flag}_{args.top_h}_{args.p}_{args.q}_{args.max_p}"
            )
            if args.heads_random_seed != -1:
                lvlm_prune_path = os.path.join(
                    lvlm_prune_path,
                    f"{args.heads_random_seed}"
                )
        print(lvlm_prune_path)
        model.save_pretrained(lvlm_prune_path)
        tokenizer.save_pretrained(lvlm_prune_path)
        processor.save_pretrained(lvlm_prune_path)
    return

if __name__ == "__main__":
    main()
