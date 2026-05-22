import argparse

from character_bert.pretraining.utils.paths import ensure_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train CharacterBERT.")
    parser.add_argument(
        "--output-dir",
        default="results/pretraining",
        help="Directory where pretraining artifacts will be written.",
    )
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)
    raise NotImplementedError(
        "Pretraining is being rebuilt in the refreshed app structure. "
        f"Output directory is ready at: {output_dir}"
    )
