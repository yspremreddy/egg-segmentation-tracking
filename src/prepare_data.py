"""Pool the Roboflow export (train/valid/test folders) into data/images/ with clean names.

Roboflow's own split is ignored: Task 2 needs its own dev/eval split (see make_split.py).
Input : egg.v1i.folder/{train,valid,test}/egg/not_damaged_<N>_jpg.rf.<hash>.jpg
Output: data/images/not_damaged_<N>.jpg  (110 images)
Usage : python src/prepare_data.py [path/to/egg.v1i.folder]
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "egg.v1i.folder"
dst = ROOT / "data" / "images"
dst.mkdir(parents=True, exist_ok=True)

pat = re.compile(r"^(?P<stem>.+?)_jpg\.rf\.[0-9a-f]+\.jpg$")
n, seen = 0, set()
for p in sorted(src.rglob("*.jpg")):
    m = pat.match(p.name)
    if not m:
        print("skipping (unexpected name):", p)
        continue
    name = m["stem"] + ".jpg"
    if name in seen:
        raise SystemExit(f"filename collision: {name}")
    seen.add(name)
    shutil.copy2(p, dst / name)
    n += 1
print(f"copied {n} images to {dst}")
