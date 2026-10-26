#!/bin/sh
# Rebuild UID 62's (e45s3) compiled kernels from the sources in this folder and compare them with the files in the committed
# image: the manifest is read by digest, layers 16 and 17 are unpacked after a sha256 check of each blob, each binary's layer
# copy is checked, and every ELF file in layers 16-17 must be one of the rebuilt binaries.
# usage: ./verify.sh <nvcc 13.4>      needs: crane, tar, sha256sum, od, docker (for ftokc.so: ubuntu:24.04 gcc 13.3)
set -e
NVCC134=${1:?path to nvcc 13.4}
case $NVCC134 in
  /*) ;;
  */*) NVCC134=$PWD/$NVCC134;;
  *) NVCC134=$(command -v "$NVCC134") || { echo "nvcc not found: $1"; exit 1; };;
esac
"$NVCC134" --version | tail -n 2
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=ghcr.io/masterdev777/s1r
REF=$IMG@sha256:8467e642812347a6c34f5c3f26eb7dd42678e4a4a532aad09f3d22d9fa0138a7
BINS="ngemm5.cubin ngemm9b.cubin ngemm9b_wn.cubin ngemm9b_wnt.cubin ngemm9t.cubin ngemm9t_wn.cubin ngemm9t_wnt.cubin \
ngemm9l.cubin ngr.cubin ngr_wn.cubin xa.cubin kkt2n.cubin ffs.cubin pron.cubin onorm.cubin kf.cubin \
kf_wn.cubin kf_wnt.cubin skx6.cubin skx6t.cubin rn.cubin ngemm7.cubin kfx.cubin zcin.cubin xa3.cubin \
onormpv.cubin ngemm9b_trow.cubin ngemm9t_trow.cubin ngemm7_trow.cubin ngr_trow.cubin ngr_wnt.cubin \
kf_trow.cubin kfx_trow.cubin ngemm8.cubin ngemm7n.cubin ngemm7w.cubin ngemm7wn.cubin q8.cubin q8k.cubin \
q8pb.cubin q8f2.cubin q8ri.cubin q8rf.cubin onormpvq.cubin onormpvqn.cubin q8gq.cubin q8h2.cubin ftokc.so"
W=$(mktemp -d)
MADE=""   # folders build_inplace created under /opt (removed again on exit)
trap 'rm -rf "$W"; for d in $MADE; do rmdir "$d" 2>/dev/null || true; done' EXIT
FAIL=0; NCOPY=0; NSAME=0; NELF=0
crane manifest "$REF" > "$W/manifest.json" 2>/dev/null || true
[ "sha256:$(sha256sum < "$W/manifest.json" | cut -d' ' -f1)" = "${REF#*@}" ] ||
  { echo "cannot read the manifest of $REF (crane / network?)"; exit 1; }
[ "$(grep -o 'sha256:[0-9a-f]\{64\}' "$W/manifest.json" | wc -l)" -eq 18 ] ||
  { echo "$REF does not have 17 layers"; exit 1; }
RT="sha256:41140838cd967104f816a16f4132acefcc18315b8585b5918eaed7dbf9e164c1
sha256:794c52e812b3bc4775dfd0ead04207d6ce98b422d61b32a0b52806bf1626985e
sha256:f4d2483a1161e0819610d3d675d7b81aa685c0f0e5e21eaf2da47c8ce92f58d2
sha256:bcc8fa79bf4cf0a65041e3a34b6b32e14966c0074879c370adc01a51cc8e0cea
sha256:e74503aa74b89a0672f0692f38a476994c4ce4ac1abba10d345a323a78a2a821
sha256:5e534237acda9cd336ace46c8bcf53303b200d6c70ce30a5e0346a479fc0e467
sha256:5f9728c239eea0750cda8725b189a498f1fd7e91491652ed97753f87a53cdd30
sha256:415e807e3ba1d5e37088342dbbaa0b15514f759b0a6b73f00b9c26be6d69ea33
sha256:ec1101e9317f072ac2baf9f43b520b3515b68d3573bc9720bd124621c3c83434
sha256:512cef068237d2bb765c52e8c811d80566c4d2b7f6916c04552b40d5b1f6df48
sha256:7124aa5c534f67fa2e5310e67db171481d28093022be8fb464a51fab5f882fef
sha256:82755c54b1adaea97b7950063722e05fda99e3a6302a5f4bbdcbe825a8122e41
sha256:e50138bd3125f743baae54c8802f35cb244e2bec5b9256d9feea5c5d7e99fe50
sha256:fa0077ecac0e5cea6a1186d178d2a022346411fafe7fb5d7c0a2ca364b2722f7
sha256:c1e007f2186caa28a555a15e7572c5219e454243abcc43d5fcbd53681db78990"
i=0
for d in $RT; do  # layers 1-15 = the official runtime image (ghcr.io/reaxlabs/reax-s1-fast-runtime:1.2.0-rc1, sha256:d857a90eaf168d1d...)
  i=$((i + 1))
  [ "$(grep -o 'sha256:[0-9a-f]\{64\}' "$W/manifest.json" | sed -n "$((i + 1))p")" = "$d" ] ||
    { echo "layer $i of $REF is not the official runtime layer $d"; exit 1; }
done
layer() {  # layer <n> <digest>: the manifest's n-th layer; check the blob's sha256 and unpack it into $W/L<n>
  [ "$(grep -o 'sha256:[0-9a-f]\{64\}' "$W/manifest.json" | sed -n "$(($1 + 1))p")" = "$2" ] ||
    { echo "layer $1 of $REF is not $2"; exit 1; }
  crane blob "$IMG@$2" > "$W/blob" 2>/dev/null || true
  [ "sha256:$(sha256sum < "$W/blob" | cut -d' ' -f1)" = "$2" ] || { echo "layer $1 ($2) did not download (crane / network?)"; exit 1; }
  mkdir -p "$W/L$1"
  tar -xzf "$W/blob" -C "$W/L$1" || { echo "layer $1 ($2) did not unpack"; exit 1; }
  rm -f "$W/blob"
  [ -n "$(ls -A "$W/L$1/opt/s1eng" 2>/dev/null)" ] || { echo "layer $1 ($2) did not unpack"; exit 1; }
}
layer 16 sha256:585c22a8c1b0917c0a66de67685668fa58d4d8b817d614385e522f7213aa9d5f
layer 17 sha256:f92353e95de6dd7356928de1873df11bd900cc0859d289dbe6e5bab953907ad5
mkdir -p "$W/b" && cd "$W/b" && cp "$HERE"/*.cu "$HERE"/*.c "$HERE"/*.h .
build() {  # build <nvcc> <log> <args...>
  nv=$1; log=$2; shift 2
  if ! "$nv" "$@" > "$log" 2>&1; then echo "build failed: $nv $*"; tail -20 "$log"; exit 1; fi
}
build_inplace() {  # build_inplace <nvcc> <binary> <args>: byte-identical only from /opt/s1eng/s1eng (path CRC in a symbol)
  if [ -z "$(ls -A /opt/s1eng/s1eng 2>/dev/null)" ]; then   # missing or empty: usable, and nothing of anyone else's is there
    [ -e /opt/s1eng ] || MADE="/opt/s1eng/s1eng /opt/s1eng"
    [ -e /opt/s1eng/s1eng ] || [ -n "$MADE" ] || MADE=/opt/s1eng/s1eng
    if mkdir -p /opt/s1eng/s1eng 2>/dev/null && [ -w /opt/s1eng/s1eng ]; then
      cp "$HERE"/*.cu "$HERE"/*.h /opt/s1eng/s1eng/
      if (cd /opt/s1eng/s1eng && "$1" $3 > "$W/b/$2.log" 2>&1); then cp "/opt/s1eng/s1eng/$2" "$W/b/$2"; ok=1; else ok=0; fi
      for f in "$HERE"/*.cu "$HERE"/*.h; do rm -f "/opt/s1eng/s1eng/${f##*/}"; done
      rm -f "/opt/s1eng/s1eng/$2"
      [ "$ok" = 1 ] || { echo "build failed (in place): $1 $3"; tail -20 "$W/b/$2.log"; exit 1; }
      return 0
    fi
  fi
  echo "($2: /opt/s1eng/s1eng not usable here (not empty, or not creatable); built in the scratch folder: only .strtab / symbol tables differ, expect DIFFERS)"
  "$1" $3 > "$2.log" 2>&1 || { echo "build failed: $1 $3"; tail -20 "$2.log"; exit 1; }
}
build_gcc() {  # ftokc.so with Ubuntu 24.04's gcc 13.3
  docker run --rm --platform linux/amd64 -v "$W/b:/w" -w /w ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac sh -c \
    "apt-get update -qq >/dev/null && apt-get install -y -qq gcc-13=13.3.0-6ubuntu2~24.04.1 >/dev/null 2>&1 && ln -sf /usr/bin/gcc-13 /usr/local/bin/gcc && $1" ||
    { echo "ftokc.so: the build in docker failed (docker / apt / the gcc-13 pin 13.3.0-6ubuntu2~24.04.1)"; exit 1; }
}
check() {  # check <name>: the rebuilt file against every layer that has it
  [ -f "$1" ] || { echo "$1: not built"; FAIL=1; return 0; }
  a=$(sha256sum "$1" | cut -d' ' -f1); k=0
  for n in 17 16; do
    f="$W/L$n/opt/s1eng/s1eng/$1"; [ -f "$f" ] || continue; k=$((k + 1)); NCOPY=$((NCOPY + 1))
    b=$(sha256sum "$f" | cut -d' ' -f1)
    if [ "$a" = "$b" ]; then echo "layer $n $1: byte-identical ($a)"; NSAME=$((NSAME + 1)); else echo "layer $n $1: DIFFERS (rebuilt $a, image $b)"; FAIL=1; fi
  done
  [ "$k" -gt 0 ] || { echo "$1: in no layer of the image"; FAIL=1; }
}
build "$NVCC134" ngemm5.cubin.log -arch=sm_120a -O3 -cubin -o ngemm5.cubin ngemm5.cu
build "$NVCC134" ngemm9b.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9b.cubin ngemm9b.cu
build "$NVCC134" ngemm9b_wn.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngemm9b_wn.cubin ngemm9b.cu
build "$NVCC134" ngemm9b_wnt.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngemm9b_wnt.cubin ngemm9b.cu
build "$NVCC134" ngemm9t.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9t.cubin ngemm9t.cu
build "$NVCC134" ngemm9t_wn.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngemm9t_wn.cubin ngemm9t.cu
build "$NVCC134" ngemm9t_wnt.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngemm9t_wnt.cubin ngemm9t.cu
build "$NVCC134" ngemm9l.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9l.cubin ngemm9l.cu
build "$NVCC134" ngr.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ngr.cubin ngr.cu
build "$NVCC134" ngr_wn.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngr_wn.cubin ngr.cu
build "$NVCC134" xa.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o xa.cubin xa.cu
build "$NVCC134" kkt2n.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o kkt2n.cubin kkt2n.cu
build "$NVCC134" ffs.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o ffs.cubin ffs.cu
build "$NVCC134" pron.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DPRON_SHIP -o pron.cubin pron.cu
build_inplace "$NVCC134" onorm.cubin '-O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onorm.cubin onorm.cu'
build "$NVCC134" kf.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o kf.cubin kf.cu
build "$NVCC134" kf_wn.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DKF_WTL -DKF_NMAJ -o kf_wn.cubin kf.cu
build "$NVCC134" kf_wnt.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DKF_WTL -DKF_NMAJ -DTROW -o kf_wnt.cubin kf.cu
build "$NVCC134" skx6.cubin.log -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6.cubin skx6.cu
build "$NVCC134" skx6t.cubin.log -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6t.cubin skx6t.cu
build "$NVCC134" rn.cubin.log -cubin -arch=sm_120a -O3 -fmad=false -Xptxas -v -o rn.cubin rn.cu
build "$NVCC134" ngemm7.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -o ngemm7.cubin ngemm7.cu
build "$NVCC134" kfx.cubin.log -arch=sm_120a -O3 -cubin -Xptxas -v -o kfx.cubin kfx.cu
build "$NVCC134" zcin.cubin.log -cubin -arch=sm_120a -O3 -Xptxas -v -o zcin.cubin zcin.cu
build "$NVCC134" xa3.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o xa3.cubin xa3.cu
build_inplace "$NVCC134" onormpv.cubin '-O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onormpv.cubin onormpv.cu'
build "$NVCC134" ngemm9b_trow.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngemm9b_trow.cubin ngemm9b.cu
build "$NVCC134" ngemm9t_trow.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngemm9t_trow.cubin ngemm9t.cu
build "$NVCC134" ngemm7_trow.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -DTROW -o ngemm7_trow.cubin ngemm7.cu
build "$NVCC134" ngr_trow.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngr_trow.cubin ngr.cu
build "$NVCC134" ngr_wnt.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngr_wnt.cubin ngr.cu
build "$NVCC134" kf_trow.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o kf_trow.cubin kf.cu
build "$NVCC134" kfx_trow.cubin.log -arch=sm_120a -O3 -cubin -Xptxas -v -DTROW -o kfx_trow.cubin kfx.cu
build "$NVCC134" ngemm8.cubin.log -O3 -arch=sm_120a -cubin -o ngemm8.cubin ngemm8.cu
build "$NVCC134" ngemm7n.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -o ngemm7n.cubin ngemm7n.cu
build "$NVCC134" ngemm7w.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -DNG7_WTL -o ngemm7w.cubin ngemm7w.cu
build "$NVCC134" ngemm7wn.cubin.log -cubin -O3 -std=c++17 -arch=sm_120a -DNG7_WTL -o ngemm7wn.cubin ngemm7wn.cu
build "$NVCC134" q8.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8.cubin q8.cu
build "$NVCC134" q8k.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8k.cubin q8k.cu
build "$NVCC134" q8pb.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8pb.cubin q8pb.cu
build "$NVCC134" q8f2.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8f2.cubin q8f2.cu
build "$NVCC134" q8ri.cubin.log -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o q8ri.cubin q8ri.cu
build "$NVCC134" q8rf.cubin.log -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o q8rf.cubin q8rf.cu
build_inplace "$NVCC134" onormpvq.cubin '-O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -DONORM_Q8 -o onormpvq.cubin onormpv.cu'
build_inplace "$NVCC134" onormpvqn.cubin '-O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -DONORM_Q8 -DONORM_Q8_NOO -o onormpvqn.cubin onormpv.cu'
build "$NVCC134" q8gq.cubin.log -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o q8gq.cubin q8gq.cu
build "$NVCC134" q8h2.cubin.log -O3 -arch=sm_120a -cubin -Xptxas -v -o q8h2.cubin q8h2.cu
build_gcc 'gcc -O2 -fPIC -shared -nostdlib -ffreestanding -fno-builtin -o ftokc.so ftokc.c'
check ngemm5.cubin
check ngemm9b.cubin
check ngemm9b_wn.cubin
check ngemm9b_wnt.cubin
check ngemm9t.cubin
check ngemm9t_wn.cubin
check ngemm9t_wnt.cubin
check ngemm9l.cubin
check ngr.cubin
check ngr_wn.cubin
check xa.cubin
check kkt2n.cubin
check ffs.cubin
check pron.cubin
check onorm.cubin
check kf.cubin
check kf_wn.cubin
check kf_wnt.cubin
check skx6.cubin
check skx6t.cubin
check rn.cubin
check ngemm7.cubin
check kfx.cubin
check zcin.cubin
check xa3.cubin
check onormpv.cubin
check ngemm9b_trow.cubin
check ngemm9t_trow.cubin
check ngemm7_trow.cubin
check ngr_trow.cubin
check ngr_wnt.cubin
check kf_trow.cubin
check kfx_trow.cubin
check ngemm8.cubin
check ngemm7n.cubin
check ngemm7w.cubin
check ngemm7wn.cubin
check q8.cubin
check q8k.cubin
check q8pb.cubin
check q8f2.cubin
check q8ri.cubin
check q8rf.cubin
check onormpvq.cubin
check onormpvqn.cubin
check q8gq.cubin
check q8h2.cubin
check ftokc.so
find "$W"/L* -type f > "$W/files.txt"
while read -r f; do  # every ELF file in layers 16-17 must be one of the binaries rebuilt above, in /opt/s1eng/s1eng
  [ "$(head -c 4 "$f" | od -An -tx1 | tr -d ' \n')" = 7f454c46 ] || continue
  NELF=$((NELF + 1)); r=${f#"$W"/L*/}; n=${r##*/}
  case " $BINS " in *" $n "*) [ "$r" = "opt/s1eng/s1eng/$n" ] && continue;; esac
  echo "compiled file not rebuilt here: $f"; FAIL=1
done < "$W/files.txt"
[ "$NCOPY" -eq "$NELF" ] || { echo "ELF files in layers 16-17: $NELF, checked copies: $NCOPY"; FAIL=1; }
echo "$NSAME/$NCOPY layer copies byte-identical (48 binaries; $NELF ELF files in layers 16-17)"
echo "image: $REF"
exit $FAIL
