#!/bin/bash

# === CONFIGURATION ===
INPUT_DIR="/home/vincent/Documents/micmactest"     
WORK_DIR="./micmactest"
DENSE_OUT_DIR="$WORK_DIR/DenseDepth"
WGS84_TO_UTM_XML="$INPUT_DIR/WGS84toUTM.xml"

mkdir -p "$WORK_DIR"
mkdir -p "$DENSE_OUT_DIR"
cp "$INPUT_DIR"/*.tif "$WORK_DIR"
cp "$WGS84_TO_UTM_XML" "$WORK_DIR"

cd "$WORK_DIR" || exit

# === 1. Convertir les RPC pour MicMac ===
echo "Conversion RPC -> GenBundle"
mm3d Convert2GenBundle ".*.tif" ".*.tif" RPC-d0 ChSys=$(basename "$WGS84_TO_UTM_XML") Degre=0

# === 2. Boucle sur les images ===
for img in *.tif; do
    name="${img%.tif}"
    echo ""
    echo "==== Traitement de $name ===="

    # 2.a Densification (Malt)
    mm3d Malt GeomImage ".*.tif" RPC-d0 Master="$img" SzW=1 ResolTerrain=1 EZA=1 NbVI=2 ZoomF=4 DirMEC="MM-$name/"

    # 2.b Export des cartes de profondeur
    mm3d TestLib GeoreferencedDepthMap "MM-$name" "$img" Ori-RPC-d0 OutDir="$DENSE_OUT_DIR" Mask=1 Scale=4
done

echo ""
echo "✅ Tout est terminé ! Les cartes sont dans : $DENSE_OUT_DIR"
