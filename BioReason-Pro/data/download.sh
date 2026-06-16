#!/bin/bash

export SCRATCH="${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}"
DEST="$SCRATCH"
mkdir -p "$DEST"

# Download from Google Drive folders using gdown
# Folder 1: https://drive.google.com/drive/folders/1Frm1o4zzFOpnA8n8GdJieYAt57poLpxk
gdown --folder "https://drive.google.com/drive/folders/1Frm1o4zzFOpnA8n8GdJieYAt57poLpxk" -O "$DEST/protein" --remaining-ok

# Folder 2 (genomics, DNA): https://drive.google.com/drive/folders/1lz00_m0mASxLggnPtBCtxJQ3hvMeS5XP
gdown --folder "https://drive.google.com/drive/folders/1lz00_m0mASxLggnPtBCtxJQ3hvMeS5XP" -O "$DEST/genomics" --remaining-ok

echo "Download complete. Files saved to $DEST"
