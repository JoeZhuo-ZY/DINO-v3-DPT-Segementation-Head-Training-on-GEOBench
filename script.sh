tasks=("chesapeake" "neontree" "nzcattle" "sacrop" "cashew" "pv4ger_seg")
for task in "${tasks[@]}"; do
    python train.py \
        --dataconfig-path /home/user/DINOv3/dinov3/seg_head/geobench_config/ \
        --task "$task" \
        --learning-rate 3e-5 \
        --batch-size 8
done
