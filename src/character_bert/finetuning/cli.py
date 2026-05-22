import argparse

from character_bert.finetuning.utils.paths import ensure_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CharacterBERT.")
    parser.add_argument(
        "--output-dir",
        default="results/finetuning",
        help="Directory where fine-tuning artifacts will be written.",
    )
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)
    raise NotImplementedError(
        "Fine-tuning is being rebuilt in the refreshed app structure. "
        f"Output directory is ready at: {output_dir}"
    )
