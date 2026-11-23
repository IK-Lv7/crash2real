#!/usr/bin/env bash
# From a trained scene to finished cases: search -> render + labels -> Difix refinement -> preview GIF.
# Usage (repository root): bash scripts/run_scene.sh <scene_idx> [rear|cutin|both]
# Output: outputs/scene<N>/cases_{rear,cutin}.json and outputs/scene<N>/<case>/ (see crash2real/generate.py),
#         outputs/scene<N>/gifs/<case>.gif. Finished cases are skipped. About 10 min per case.
# Partners on the left are rendered with cameras 0 1 3, on the right with 0 2 4.
# Set REFINE=0 to skip Difix (then the GIFs show the raw renders). Always look at the frames before using a case.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
i="${1:?scene_idx}"; kinds="${2:-both}"; [ "$kinds" = both ] && kinds="rear cutin"
name=$(sed -n "$((i+1))p" configs/av2_logs.txt | cut -c1-8)
RUN="outputs/av2/${name}_omnire"; OUT="outputs/scene$i"
[ -f "$RUN/checkpoint_final.pth" ] || { echo "train scene $i first (scripts/train_scene.sh $i)"; exit 1; }

conda activate drivestudio
for k in $kinds; do
  [ -f "$OUT/cases_$k.json" ] || python -m crash2real.search "$k" --scene_idx "$i" --run "$RUN" --out "$OUT/cases_$k.json"
done
python3 - $(for k in $kinds; do echo "$OUT/cases_$k.json"; done) > "$OUT/.todo" <<'PY'
import json, sys
for f in sys.argv[1:]:
    for c in json.load(open(f))["cases"]:
        print(c["name"], "0 1 3" if c["side"] == "L" else "0 2 4", c["cli"], sep="\t")
PY
while IFS=$'\t' read -r -u 3 case cams cli; do
  dir="$OUT/$case"; last=${cams##* }
  if [ ! -f "$dir/boxes.json" ]; then
    echo "=== $case"
    python -m crash2real.generate --run "$RUN" $cli --cams $cams --out_dir "$dir"
  fi
  if [ "${REFINE:-1}" = 1 ] && [ ! -d "$dir/refined_cam$last" ]; then
    conda activate difix
    for c in $cams; do python -m crash2real.refine --in_dir "$dir/cam$c" --out_dir "$dir/refined_cam$c"; done
    conda activate drivestudio
  fi
  python -m crash2real.make_gif "$dir" --cams 0 $(echo $cams | cut -d' ' -f2) --scale 0.4 --out "$OUT/gifs/$case.gif"
done 3< "$OUT/.todo"
rm -f "$OUT/.todo"
echo "scene $i done: $OUT"
