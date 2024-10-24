#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


cd src

python_script="main.py"

scratch=False
cuda='cuda:0'
dataset='demosath_test'
feature_num=20
seq_len=150
columns_to_mask='[0, 14,15,16,17,18,19]'
# missing_pattern='block'
# missing_ratio=0.4
# val_missing_ratio=0.4
# test_missing_ratio=0.4

dataset_path="../datasets/$dataset/"
checkpoint_path="../saved_models/demosath/block/0.0033/model_2024-10-16-11-31-02.pth"

if [ $scratch = True ]; then
    log_path="../logs/scratch"
else
    log_path="../logs/test"
fi

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "Folder created: $log_path"
else
    echo "Folder already exists: $log_path"
fi

for ((i=1; i<=10; i++))
do
    seed=$i

    echo "Running iteration $i with seed $seed on device $cuda"

    if [ $scratch = True ]; then
        echo "did not run. Change scratch parameter to False."
    
    else
        nohup python -u $python_script \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path $dataset_path \
            --seq_len $seq_len \
            --feature $feature_num \
            --ratio_mask $ratio_mask \
            --checkpoint_path $checkpoint_path \
            --nsample 100 \
            # --missing_pattern $missing_pattern \
            # --missing_ratio $missing_ratio \
            # --val_missing_ratio $val_missing_ratio \
            # --test_missing_ratio $test_missing_ratio \

            > $log_path/${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}_virt_moor_noWindSpeed.log 2>&1 &
    fi

    wait

    echo ""
done
