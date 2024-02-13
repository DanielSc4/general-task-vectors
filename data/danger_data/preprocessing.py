import json
import fire
import os

# standard format:
# [
#     {
#         'input': str,
#         'output': str,
#     },
# ]


def load_json_dataset(json_path):
    with open(json_path, encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

def main():
    files_in_current_dir = os.listdir('.')
    for file in fi

    load_json_dataset()








if __name__ == "__main__":
    fire.Fire(main)
