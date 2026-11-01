#!/usr/bin/env python3
"""Portable CLI for the existing blur -> downsample -> noise implementation."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "--input", required=True, type=Path,
                        help="Directory containing HR source images (non-recursive).")
    parser.add_argument("--output-dir", "--output", required=True, type=Path,
                        help="Directory for LR images with the original filenames.")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("degradation_default.json"),
                        help="JSON overrides for the original degradation settings.")
    parser.add_argument("--seed", type=int, help="Override the seed in the JSON config.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing existing output images.")
    args = parser.parse_args(argv)
    try:
        if __package__:
            from . import tip_make_LR as degradation
        else:
            import tip_make_LR as degradation
        cfg = degradation.CONFIG.copy()
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("The degradation config must be a JSON object.")
        allowed = set(cfg) - {"input_dir", "output_dir"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(settings)
        if args.seed is not None:
            cfg["seed"] = args.seed
        if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32 - 1].")
        input_dir = args.input_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        if not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        if input_dir == output_dir:
            raise ValueError("Input and output directories must be different.")
        paths = degradation.list_images(str(input_dir))
        if not paths:
            raise ValueError(f"No supported images found in {input_dir}")
        if not args.overwrite:
            conflicts = [p for p in paths if (output_dir / Path(p).name).exists()]
            if conflicts:
                raise ValueError(f"{len(conflicts)} output image(s) already exist; use --overwrite.")
        cfg["input_dir"] = str(input_dir)
        cfg["output_dir"] = str(output_dir)
        degradation.process_folder(cfg)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
