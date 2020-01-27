import scraper_openlibrary as s
import os
import json
import argparse


def main(opts):

    # read json file
    json_file = opts.dataset_json
    with open(json_file, 'r') as f:
        illustrator_dataset = json.load(f)

    print(json.dumps(illustrator_dataset, indent=4, sort_keys=True))

    # init helper
    olh = s.OpenLibHelper(opts.openlib_username, opts.openlib_password)

    for illustrator in illustrator_dataset:
        dir_name = os.path.join(opts.download_dir, illustrator)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # search for an illustrator and books
        lower_case_list = [x.lower() for x in illustrator_dataset[illustrator]]
        if opts.download_json:
            olh.search_author(illustrator, dir_name, lower_case_list)
        else:
            olh.search_author(illustrator, dir_name, [])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_json', type=str, default="dataset.json",
                        help='dataset json file')
    parser.add_argument('--download_json', type=bool, default="True",
                        help='if True download only books in the json, else download all books for an illustrator')
    parser.add_argument('--openlib_username', type=str, default="",
                        help='username of openlib account')
    parser.add_argument('--openlib_password', type=str, default="",
                        help='password of openlib account')
    parser.add_argument('--download_dir', default='.',
                        help='where to download dataset')

    opts = parser.parse_args()

    main(opts)
