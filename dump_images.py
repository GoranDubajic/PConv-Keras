import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np

from pathlib import Path

from libs.util import MaskGenerator

from main import AugmentingDataGenerator



def main():
    parser = argparse.ArgumentParser(
        description="Save input images as NPZ."
    )
    parser.add_argument("input_dir", help="Input directory with RGB and silhouette images.")
    parser.add_argument("output_dir", help="Output directory.")
    args = parser.parse_args()
    data_datagen = AugmentingDataGenerator(rescale=1./255)
    data_generator = data_datagen.flow_from_directory(
        args.input_dir,
        MaskGenerator(512, 512, 3),
        target_size=(512, 512),
        batch_size=1,
        shuffle=False)

    count = 0
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for data in data_generator:
        (masked, mask), original = data
        npz_path = str(out_path.joinpath(f"{count:04}"))
        np.savez_compressed(npz_path, original=original[0, :, :, :])
        print(f"Saved original image to: {npz_path}")
        count += 1
        if count > 52:
            break


if __name__ == "__main__":
    main()

