# source .venv/bin/activate


(yes | python -m main \
    --model_name stabilityai/stablelm-2-zephyr-1_6b \
    --dataset_name XS_test \
    --mean_support 70 \
    --aie_support 20 \
    --icl_examples 4 \
    --batch_size 1 \
    --load_in_8bit \
    --use_local_backups \
)

