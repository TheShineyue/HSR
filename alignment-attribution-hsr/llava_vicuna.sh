model="llava-next-vicuna-hf-7b"
method="lvlm_wanda_hf"
# method="lvlm_wanda_recover_heads"
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
    --data_mode $data_mode 
    # --p 0.5 --q 0.4 --max_p 0.7 --top_h 10 --heads_paths $heads_paths --only_heads 
    # --dump_wanda_score