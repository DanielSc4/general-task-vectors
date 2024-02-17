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




(yes | python -m main --model_name stabilityai/stablelm-2-zephyr-1_6b --dataset_name antonym --icl_examples 10 --batch_size 18 --load_in_8bit)

# (yes | python -m main \
#     --model_name stabilityai/stablelm-2-zephyr-1_6b \
#     --dataset_name joined-DX \
#     --multi_token_generation \
#     --mean_support 100 \
#     --aie_support 25 \
#     --icl_examples 0 \
#     --batch_size 17 \
#     --load_in_8bit \
# )

