# general-task-vectors


Run:
```bash
python -m main --model_name microsoft/phi-2 --dataset_name antonym --icl_examples 4 --batchsize 32
```

```bash
(yes | nohup python -m main --model_name stabilityai/stablelm-2-zephyr-1_6b --dataset_name antonym --icl_examples 10 --batch_size 18 --load_in_8bit --use_local_backups) > logs/output_log.out &
```