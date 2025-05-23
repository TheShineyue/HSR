# [Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models](https://arxiv.org/abs/2505.16104)

Yue Li*, Xin Yi*, Dongsheng Shi, Gerard de Melo, Xiaoling Wang, Linlin Wang†.

[![arXiv](https://img.shields.io/badge/arXiv-2505.14679-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2505.16104) 

![hsr.png](resource/hsr.png)

## 🔔 News

- \[2025.05.23\]. We posted the preprint on arxiv. 
- \[2025.05.21\]. We open sourced the code of our project.
- \[2025.05.16\]. 🎉 Our work is accepted by **ACL 2025 (Findings)**! 

## 🚀 Quick Start

We use [llava-next-vicuna-7b-hf](https://modelscope.cn/models/llava-hf/llava-v1.6-vicuna-7b-hf) as an example to show the workflow of hsr.

### ⚙️ Environment Preparation

```python
conda create --name hsr python=3.9
conda activate hsr
pip install -r requirements.txt
```

### 📂 Dataset Preparation

We utilize the [VLGuard](https://huggingface.co/datasets/ys-zong/VLGuard) to identify safety-critical attention heads and neurons. The training subset is downloaded into the "data_process" folder.

Next, we execute the get_data.py script, which generates two files: train_safe_safes.json and train_unsafes.json. These files serve as the utility and safety datasets, respectively.

### 🎬 HSR

#### Step 1: Identifying Safety-Critical Heads

In the SafetyHeadAttribution-hsr folder, modify the model path, data path, and other relevant settings in llava_next_ships.py, then execute the script. By default, the files containing the Ships scores for each head will be saved in the SafetyHeadAttribution-hsr/exp_res/llava-next-vicuna directory.

#### Step 2: Identifying Safety-critical Neurons

In the alignment-attribution-hsr folder, modify the llama3_llava.sh script. In addition, you need to modify the settings (e.g., model path) in main.py and lib/data.py.

- get pruned model
```
model="llava-next-vicuna-hf-7b"
method="lvlm_wanda_hf"
type="unstructured"
device="cuda:7"
suffix="weightonly"
data_mode="train_safes"
heads_paths="/home/liyue/projects/hsr/psafety/SafetyHeadAttribution-hsr/exp_res/llava-next-vicuna/train_unsafes.json_0.jsonl"
save_dir="out/$model/$type/${method}_${suffix}/"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data VLguard \
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --save $save_dir \
    --device $device \
    --data_mode $data_mode 
```

- get important score
```
# you shold get safety and utility important scores
model="llava-next-vicuna-hf-7b"
method="lvlm_wanda_hf"
type="unstructured"
device="cuda:7"
suffix="weightonly"
data_mode="train_safes" # train_unsafes
heads_paths="/home/liyue/projects/hsr/psafety/SafetyHeadAttribution-hsr/exp_res/llava-next-vicuna/train_unsafes.json_0.jsonl"
save_dir="out/$model/$type/${method}_${suffix}/"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data VLguard \
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --save $save_dir \
    --device $device \
    --data_mode $data_mode --dump_wanda_score
```
- get realigned model
```
model="llava-next-vicuna-hf-7b"
method="lvlm_wanda_recover_heads"
type="unstructured"
device="cuda:0"
suffix="weightonly"
data_mode="train_safes"
heads_paths="/home/liyue/projects/hsr/psafety/SafetyHeadAttribution-hsr/exp_res/llava-next-vicuna/train_unsafes.json_0.jsonl"
save_dir="out/$model/$type/${method}_${suffix}/"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data VLguard \
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --save $save_dir \
    --device $device \
    --data_mode $data_mode --p 0.5 --q 0.5 --max_p 0.7 --top_h 10 --heads_paths $heads_paths
```

### 📜 Citation
If you find our work useful, please consider citing our paper:
```
@misc{li2025hierarchicalsafetyrealignmentlightweight,
      title={Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models}, 
      author={Yue Li and Xin Yi and Dongsheng Shi and Gerard de Melo and Xiaoling Wang and Linlin Wang},
      year={2025},
      eprint={2505.16104},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.16104}, 
}
```

Our codebase is built upon on the following works:
```
@article{zhou2024role,
  title={On the Role of Attention Heads in Large Language Model Safety},
  author={Zhou, Zhenhong and Yu, Haiyang and Zhang, Xinghua and Xu, Rongwu and Huang, Fei and Wang, Kun and Liu, Yang and Fang, Junfeng and Li, Yongbin},
  journal={arXiv preprint arXiv:2410.13708},
  year={2024}
}
```
```
@inproceedings{wei2024assessing,
  title={Assessing the brittleness of safety alignment via pruning and low-rank modifications},
  author={Wei, Boyi and Huang, Kaixuan and Huang, Yangsibo and Xie, Tinghao and Qi, Xiangyu and Xia, Mengzhou and Mittal, Prateek and Wang, Mengdi and Henderson, Peter},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={52588--52610},
  year={2024}
}
```
