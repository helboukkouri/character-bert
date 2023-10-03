# Adapted from:
# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

""" Download pre-trained models from Google drive. """
import os
import gdown
import argparse
import tarfile
import logging
import requests

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO)

MODEL_TO_URL = {
    'general_character_bert': 'https://drive.google.com/uc?id=11-kSfIwSWrPno6A4VuNFWuQVYD8Bg_aZ',
    'medical_character_bert': 'https://drive.google.com/uc?id=1LEnQHAqP9GxDYa0I3UrZ9YV2QhHKOh2m',
    'general_bert': 'https://drive.google.com/uc?id=1fwgKG2BziBZr7aQMK58zkbpI0OxWRsof',
    'medical_bert': 'https://drive.google.com/uc?id=1GmnXJFntcEfrRY4pVZpJpg7FH62m47HS',
}

def download_file_from_google_drive(url, destination):
    gdown.download(url, destination, quiet=False)

def download_model(name):
    if os.path.exists(os.path.join('pretrained-models', name)):
        logging.info(f"Path {os.path.join('pretrained-models', name)} already exists.")
        logging.info(f'Skipped download of {name} model.')
    else:
        os.makedirs(os.path.join('pretrained-models', name), exist_ok=False)
        if name == 'bert-base-uncased':
            urls = {
                'model': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin',
                'vocabulary': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt',
                'config': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json'
            }
            logging.info(f'Downloading {name} model (~420MB folder)')
            for _, url in urls.items():
                file_name = os.path.basename(url).split('-')[-1]
                file_destination = os.path.join('pretrained-models', name, file_name)
                response = requests.get(url)
                with open(file_destination, mode='wb') as f:
                    f.write(response.content)
        else:
            file_destination = os.path.join('pretrained-models', 'model.tar.xz')
            model_url = MODEL_TO_URL[name]

            logging.info(f'Downloading {name} model (~200MB tar.xz archive)')
            download_file_from_google_drive(url=model_url, destination=file_destination)

            logging.info('Extracting model from archive (~420MB folder)')
            tar = tarfile.open(file_destination, "r:xz")
            tar.extractall(path=os.path.dirname(file_destination))
            tar.close()

            logging.info('Removing archive')
            os.remove(file_destination)
        logging.info('Done.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_TO_URL.keys()) + ['bert-base-uncased', 'all'],
        help="A keyword for downloading a specific pre-trained model"
    )
    args = parser.parse_args()

    if args.model == 'all':
        for model in list(MODEL_TO_URL.keys()) + ['bert-base-uncased']:
            download_model(name=model)
    else:
        download_model(name=args.model)

if __name__ == "__main__":
    main()
