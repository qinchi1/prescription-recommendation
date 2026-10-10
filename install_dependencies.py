import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target")
    args = parser.parse_args()
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "numpy>=1.26",
        "pandas>=2",
        "torch>=2.6",
        "torch-geometric==2.6.1",
        "transformers>=4.57,<5",
        "scikit-learn>=1.3",
        "openpyxl>=3.1",
        "safetensors>=0.4",
    ]
    if args.target:
        command.extend(["--target", args.target])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
