source .venv/bin/activate

# python -m main --model_name EleutherAI/gpt-j-6b --dataset_name capitalize --icl_examples 10 --batch_size 9 --load_in_8bit --use_local_backups
# echo "country-capital"
# python -m main --model_name EleutherAI/gpt-j-6b --dataset_name country-capital --icl_examples 10 --batch_size 9 --load_in_8bit --use_local_backups
# echo "sentiment"
# python -m main --model_name EleutherAI/gpt-j-6b --dataset_name sentiment --icl_examples 10 --batch_size 9 --load_in_8bit --use_local_backups
# 
# echo "antonym"
# (yes | python -m main --model_name stabilityai/stablelm-2-zephyr-1_6b --dataset_name antonym --icl_examples 10 --batch_size 15 --load_in_8bit --use_local_backups)
# echo "capitalize"
# (yes | python -m main --model_name stabilityai/stablelm-2-zephyr-1_6b --dataset_name capitalize --icl_examples 10 --batch_size 15 --load_in_8bit --use_local_backups)
# echo "country-capitlal"
# (yes | python -m main --model_name stabilityai/stablelm-2-zephyr-1_6b --dataset_name country-capital --icl_examples 10 --batch_size 15 --load_in_8bit --use_local_backups)
# echo "sentiment"
# (yes | python -m main --model_name stabilityai/stablelm-2-zephyr-1_6b --dataset_name sentiment --icl_examples 10 --batch_size 15 --load_in_8bit --use_local_backups)




# (yes | python -m main --model_name stabilityai/stablelm-2-zephyr-1_6b --dataset_name antonym --icl_examples 10 --batch_size 18 --load_in_8bit)

# model options:
    # --model_name stabilityai/stablelm-2-zephyr-1_6b \
    # --model_name google/gemma-2b-it \


(yes | python -m eval \
    --model_name stabilityai/stablelm-2-zephyr-1_6b \
    --dataset_name joined-DX \
    --mean_activation_path output/stablelm-2-zephyr-1_6b/joined-DX_mean_activations_stabilityai-stablelm-2-zephyr-1_6b_ICL0.pt \
    --cie_path output/stablelm-2-zephyr-1_6b/joined-DX_cie_stabilityai-stablelm-2-zephyr-1_6b_ICL0.pt \
    --top_n_heads 10 \
    --eval_dim 20 \
    --load_in_8bit \
    --multi_token_evaluation \
    --corrupted_ICL False\
    --ICL_examples 0\
    --print_examples \
)


# (yes | python -m main \
#     --model_name stabilityai/stablelm-2-zephyr-1_6b \
#     --dataset_name joined-DX \
#     --multi_token_generation \
#     --mean_support 100 \
#     --aie_support 25 \
#     --icl_examples 0 \
#     --batch_size 10 \
#     --load_in_8bit \
# )

