#!/usr/bin/env bash
# Download selected Argoverse 2 Sensor logs with s5cmd (bucket path from the official user guide).
#   bash scripts/download_av2.sh list [split]             # list log ids (train | val | test, default val)
#   bash scripts/download_av2.sh get <split> <LOG_ID>...  # download logs
# Set AV2_DIR to the destination, e.g. export AV2_DIR=$HOME/data/argoverse (logs go to $AV2_DIR/sensor/<split>/<LOG_ID>).
set -euo pipefail
BUCKET="${AV2_BUCKET:-s3://argoverse/datasets/av2/sensor}"
DEST="${AV2_DIR:?set AV2_DIR to the download folder}/sensor"
S5="s5cmd --no-sign-request"
case "${1:-}" in
  list) $S5 ls "$BUCKET/${2:-val}/" ;;
  get)
    split="${2:?split (train|val|test)}"; shift 2
    [ "$#" -ge 1 ] || { echo "give at least one LOG_ID"; exit 1; }
    for log in "$@"; do
      mkdir -p "$DEST/$split/$log"; echo ">> $split/$log"
      $S5 cp "$BUCKET/$split/$log/*" "$DEST/$split/$log/"
    done ;;
  *) sed -n '2,5p' "$0"; exit 1 ;;
esac
