
# source .venv/bin/activate


echo "Running main"
python -m main \
    --model_name openai-community/gpt2 \
    --dataset_name XS_test \
    --multi_token_generation \
    --mean_support 4 \
    --aie_support 3 \
    --icl_examples 4 \
    --batch_size 1 \
    --load_in_8bit \

# echo "Running eval"
# python -m eval \
#     --model_name \
#     --dataset_name \
#     --mean_activation_path \
#     --cie_path \
#     --top_n_heads \
#     --eval_dim \
#     --load_in_8bit \
#     --print_examples \

