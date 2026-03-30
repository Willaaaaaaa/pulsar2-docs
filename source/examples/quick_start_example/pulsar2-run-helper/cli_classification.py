import argparse

from pathlib import Path

from pulsar2_run_helper.utils import Logger, sanitize

LOGGER = Logger(colors=False)


def get_parser():
    parser = argparse.ArgumentParser("CLI tools for pre-processing and post-processing.", add_help=True)

    parser.add_argument(
        "--image_path",
        type=str,
        help="The path of image file.",
    )
    parser.add_argument(
        "--axmodel_path",
        type=str,
        required=True,
        help="The path of compiled axmodel.",
    )
    parser.add_argument(
        "--intermediate_path",
        type=str,
        required=True,
        help="The path of intermediate data bin.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="The path of output files.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=224,
        help="Image size for croping (default: 224).",
    )
    parser.add_argument("--pre_processing", action="store_true", help="Do pre processing.")
    parser.add_argument("--post_processing", action="store_true", help="Do post processing.")

    return parser


def pre_processing(args):
    from pulsar2_run_helper.preprocessing import get_input_info, preprocess_classification

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Not found image file at '{image_path}'")

    axmodel_path = Path(args.axmodel_path)
    if not axmodel_path.exists():
        raise FileNotFoundError(f"Not found compiled axmodel at '{axmodel_path}'")

    img_transformed = preprocess_classification(str(image_path), crop_size=args.crop_size)
    input_names = get_input_info(axmodel_path)
    if len(input_names) != 1:
        raise NotImplementedError(f"Currently only supports length 1, but got {input_names}")

    intermediate_path = Path(args.intermediate_path)
    intermediate_path.mkdir(exist_ok=True, parents=True)

    output_path = intermediate_path / f"{sanitize(input_names[0])}.bin"
    output_path.write_bytes(img_transformed.tobytes())
    LOGGER.info(f"Write [{input_names[0]}] to '{output_path}' successfully.")


def post_processing(args):
    from pulsar2_run_helper.postprocessing import get_max_indices

    axmodel_path = Path(args.axmodel_path)
    if not axmodel_path.exists():
        raise FileNotFoundError(f"Not found compiled axmodel at '{axmodel_path}'")

    intermediate_path = Path(args.intermediate_path)
    intermediate_path.mkdir(exist_ok=True, parents=True)

    max_indices, scores = get_max_indices(axmodel_path, intermediate_path, topk=5)

    LOGGER.info("The following are the predicted score index pair.")
    for max_index, score in zip(max_indices, scores):
        LOGGER.info(f"{score:.4f}, {max_index}")


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    LOGGER.debug(f"Command Line Args: {args}")

    if args.pre_processing:
        pre_processing(args)
    if args.post_processing:
        post_processing(args)


if __name__ == "__main__":
    cli_main()
