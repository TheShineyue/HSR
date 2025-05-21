model="llama3-llava-next-8b-hf"
# method="lvlm_wanda_hf"
method="lvlm_wanda_recover_heads_gqa"
type="unstructured"
device="cuda:7"
suffix="weightonly"
data_mode="train_unsafes"
heads_paths="/home/liyue/psafety/SafetyHeadAttribution-hsr/exp_res/llama3-llava/train_unsafes.json_0.jsonl"
save_dir="out/$model/$type/${method}_${suffix}/"

python main.py \
    --model $model \
    --prune_method $method \
    --prune_data VLguard\
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --save $save_dir \
    --device $device \
    --data_mode $data_mode --p 0.5 --q 0.5 --max_p 0.7 --top_h 4 --heads_paths $heads_paths
    # --dump_wanda_score
    # --p 0.5 --q 0.5 --max_p 0.7 --top_h 4 --heads_paths $heads_paths
    