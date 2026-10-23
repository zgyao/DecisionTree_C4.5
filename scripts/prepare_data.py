#!/usr/bin/env python3
"""Verify bundled datasets offline, or regenerate from checksum-pinned sources."""
import argparse
import csv
import hashlib
import io
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/sklearn/datasets/data/"
WINE_NAMES = "alcohol malic_acid ash alcalinity_of_ash magnesium total_phenols flavanoids nonflavanoid_phenols proanthocyanins color_intensity hue od280_od315 proline".split()
CANCER_NAMES = "radius texture perimeter area smoothness compactness concavity concave_points symmetry fractal_dimension".split()
SOURCES = {
    "iris": ("iris.csv", "f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449",
             ["sepal_length", "sepal_width", "petal_length", "petal_width"]),
    "wine": ("wine_data.csv", "10e8a802908b34f86e5da8ce962f3c806694bc98450a18f61851af59f324bede", WINE_NAMES),
    "breast_cancer": ("breast_cancer.csv", "fed3eb72d0575ef6192293f5093c6e801b1476b577d0386bf4455504522172ed",
                      [f"{name}_{stat}" for stat in ["mean", "se", "worst"] for name in CANCER_NAMES]),
}
# SHA-256 of normalized .dat files, independent of network/cache availability.
NORMALIZED_HASHES = {
    "iris": "3be376aba07caba824504499743be958e10ac83d322751239ddd617c6b0df3fb",
    "wine": "2ecea4011aab0b12ad6dc25639157d8b32f21c9afedf691a87eaf1a88979935a",
    "breast_cancer": "c7fdbd88831a6343b9c637be54c65b15fc20641db589d009bdd11c4d4f06b1eb",
}


def normalize(content, names):
    rows = list(csv.reader(io.StringIO(content.decode("utf-8"))))
    count, dimensions = map(int, rows[0][:2])
    labels = rows[0][2:]
    records = [row for row in rows[1:] if row]
    if dimensions != len(names) or count != len(records):
        raise ValueError("source dimensions changed")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["Continuous"] * dimensions + ["Discrete"])
    writer.writerow(names + ["class"])
    for row in records:
        if len(row) != dimensions + 1:
            raise ValueError("source row width changed")
        index = int(row[-1])
        if not 0 <= index < len(labels):
            raise ValueError("source class index out of range")
        writer.writerow(row[:-1] + [labels[index]])
    return output.getvalue().encode("utf-8")


def verify(content, expected, name):
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="check bundled files without network access")
    mode.add_argument("--download", action="store_true", help="download verified sources and regenerate .dat files")
    args = parser.parse_args()
    for name, (filename, source_hash, names) in SOURCES.items():
        target = ROOT / "data" / f"{name}.dat"
        if args.download:
            with urllib.request.urlopen(BASE + filename, timeout=30) as response:
                content = response.read()
            verify(content, source_hash, filename)
            normalized = normalize(content, names)
            verify(normalized, NORMALIZED_HASHES[name], target.name)
            target.parent.mkdir(exist_ok=True)
            temporary = target.with_suffix(".dat.tmp")
            temporary.write_bytes(normalized)
            temporary.replace(target)
        verify(target.read_bytes(), NORMALIZED_HASHES[name], target.name)
        print(f"verified {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
