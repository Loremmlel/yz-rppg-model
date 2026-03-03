# Run from project root: python generate_proto.py
import subprocess
import sys
import re
from pathlib import Path

GENERATED_DIR = Path("src/generated")

subprocess.run([
    sys.executable, "-m", "grpc_tools.protoc",
    "-Iproto",
    f"--python_out={GENERATED_DIR}",
    f"--grpc_python_out={GENERATED_DIR}",
    "proto/frame_analysis.proto"
], check=True)

# Fix bare imports in *_grpc.py → relative imports
# grpc_tools generates "import xxx_pb2 as ..." but since the files live
# inside a package (src.generated) we need "from . import xxx_pb2 as ..."
for grpc_file in GENERATED_DIR.glob("*_pb2_grpc.py"):
    text = grpc_file.read_text(encoding="utf-8")
    fixed = re.sub(
        r"^(import )(\w+_pb2)( as )",
        r"from . import \2\3",
        text,
        flags=re.MULTILINE,
    )
    if fixed != text:
        grpc_file.write_text(fixed, encoding="utf-8")
        print(f"Fixed imports in {grpc_file}")

print("Proto files generated successfully.")

