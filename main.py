"""
Entry point. Run like:

    python main.py models/tiny_conv.onnx
"""

import sys
from frontend import pipeline

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_onnx_file>")
        sys.exit(1)

    pipeline.run(sys.argv[1])
