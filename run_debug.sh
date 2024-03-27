
# source .venv/bin/activate


# echo "Running main"
# python -m main \
#     --model_name openai-community/gpt2 \
#     --dataset_name XS_test \
#     --multi_token_generation \
#     --mean_support 4 \
#     --aie_support 3 \
#     --icl_examples 4 \
#     --batch_size 1 \
#     --load_in_8bit \

echo "Running eval"
python -m eval \
    --model_name openai-community/gpt2 \
    --dataset_name XS_test \
    --mean_activation_name mean_activations_icl4_sup4.pt \
    --cie_name cie_ICL4_sup3.pt \
    --top_n_heads 10 \
    --eval_dim 10 \
    --load_in_8bit \
    --print_examples \

