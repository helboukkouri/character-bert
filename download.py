# Adapted from:
# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

""" Download pre-trained models from Google drive. """
import os
import gdown
import argparse
import tarfile
import logging
import requests
from huggingface_hub import snapshot_download

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

HF_MODEL_TO_REPO = {
    'hf_character_bert': 'helboukkouri/character-bert',
    'hf_character_bert_medical': 'helboukkouri/character-bert-medical',
}

BERT_BASE_UNCASED_FILES = {
    'model': (
        'https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin',
        'pytorch_model.bin',
    ),
    'vocabulary': (
        'https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt',
        'vocab.txt',
    ),
    'config': (
        'https://huggingface.co/bert-base-uncased/resolve/main/config.json',
        'config.json',
    ),
}


def model_is_downloaded(path):
    return all(
        os.path.exists(os.path.join(path, filename))
        for filename in ['config.json', 'pytorch_model.bin']
    )


def download_url(url, destination):
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(destination, mode='wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract_tar_safely(tar, path):
    destination = os.path.abspath(path)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(destination, member.name))
        if os.path.commonpath([destination, member_path]) != destination:
            raise RuntimeError(f"Archive member escapes destination: {member.name}")
    tar.extractall(path=path)


def download_file_from_google_drive(url, destination):
    output = gdown.download(url, destination, quiet=False)
    if output is None or not os.path.exists(destination):
        raise RuntimeError(f"Failed to download archive from {url}")

def download_model(name):
    model_path = os.path.join('pretrained-models', name)
    if model_is_downloaded(model_path):
        logging.info(f"Path {model_path} already exists.")
        logging.info(f'Skipped download of {name} model.')
    else:
        os.makedirs(model_path, exist_ok=True)
        if name in HF_MODEL_TO_REPO:
            repo_id = HF_MODEL_TO_REPO[name]
            logging.info(f'Downloading {repo_id} from Hugging Face Hub (~730MB folder)')
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
        elif name == 'bert-base-uncased':
            logging.info(f'Downloading {name} model (~420MB folder)')
            for _, (url, file_name) in BERT_BASE_UNCASED_FILES.items():
                file_destination = os.path.join(model_path, file_name)
                if os.path.exists(file_destination):
                    logging.info(f'File {file_destination} already exists.')
                    continue
                download_url(url, file_destination)
        else:
            file_destination = os.path.join('pretrained-models', 'model.tar.xz')
            model_url = MODEL_TO_URL[name]

            logging.info(f'Downloading {name} model (~200MB tar.xz archive)')
            download_file_from_google_drive(url=model_url, destination=file_destination)

            logging.info('Extracting model from archive (~420MB folder)')
            tar = tarfile.open(file_destination, "r:xz")
            extract_tar_safely(tar, path=os.path.dirname(file_destination))
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
        choices=list(MODEL_TO_URL.keys()) + list(HF_MODEL_TO_REPO.keys()) + ['bert-base-uncased', 'all'],
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
