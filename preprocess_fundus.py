"""Crop the retinal field of view, apply CLAHE, and write a new manifest."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def process(image, size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(mask)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        image = image[y:y + h, x:x + w]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    image = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    if not {"path", "label"}.issubset(frame.columns):
        raise ValueError("manifest must contain path,label columns")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in tqdm(frame.iterrows(), total=len(frame)):
        source = Path(row.path)
        if not source.is_absolute():
            source = args.manifest.parent / source
        # imdecode/tofile work with non-ASCII Windows paths, unlike cv2.imread/imwrite.
        image = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(source)
        destination = args.output / f"{index:06d}{source.suffix.lower()}"
        extension = destination.suffix or ".png"
        ok, encoded = cv2.imencode(extension, process(image, args.size))
        if not ok:
            raise OSError(f"failed to encode {destination}")
        encoded.tofile(destination)
        rows.append({"path": destination.name, "label": int(row.label)})
    pd.DataFrame(rows).to_csv(args.output / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
