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

def main(
    json_path: str,
    output_path: str = 'data/'
):

    dataset = load_json_dataset(json_path)

    print('Printing preview of the dataset')
    for k in dataset.keys():
        print(f'\tKey: {k} [0:2]')
        print(f'\t\t {dataset[k][0:2]}')

    # instructions is the only important column
    output = [{'input': v, 'output': ''} for v in dataset['instructions']]

    # write output file in output_path
    # get dataset name from path
    dataset_name = json_path.split('/')[-1]
    

    with open(output_path + dataset_name, 'w') as fout:
        json.dump(output, fout, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
