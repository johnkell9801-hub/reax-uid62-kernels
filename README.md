# Kernel sources of the REAX s1-fast engine committed on SN92 UID 62 (e45s3)

Entry: UID 62, hotkey `5D9pojiJakcm4iLkxix3L8cVmMXKeExufbz96nBHKJ5gPWoA`, competition element
`reax-eng/1:s1-fast:ghcr.io/masterdev777/s1r@sha256:8467e642812347a6c34f5c3f26eb7dd42678e4a4a532aad09f3d22d9fa0138a7`
(engine e45s3: the e48c line with ONQ8N, derive smoothing, Q8 from layer 0, BAYES guard p* 0.20 and suffix repair from
layer 0; the int8 weights are derived at start from the public seed set, the image ships no weight file).

Every compiled part the image ships is in `/opt/s1eng/s1eng/` (48 files: 47 CUDA cubins and one host library). Each one
rebuilds **byte for byte** from the sources in this folder with the build line below, run in `/opt/s1eng/s1eng`. The
image also ships each source next to its binary, and `/opt/s1eng/s1eng/KERNELS.md` lists the same binaries with the same
binary hashes, source hashes and build lines (it also repeats 5 rows of an older table with the same binary hashes; the
repeated `kfx.cubin` / `ngemm7.cubin` rows carry the source hashes of earlier `kfx.cu` / `ngemm7.cu` versions, which
this image does not ship). 8 of the 48 binaries are new since UID 220's entry (sources published in
[reax-uid220-kernels](https://github.com/johnkell9801-hub/reax-uid220-kernels)); the other 40 are byte-identical to
files published there and are listed again so this folder is complete.

Layers: layers 1-15 are the official `ghcr.io/reaxlabs/reax-s1-fast-runtime:1.2.0-rc1` layers
(`sha256:d857a90eaf168d1d…`), unchanged. Layer 16 holds the whole `/opt/s1eng` tree (170 files, 9.4 MB gzipped) with 47
of the binaries; layer 17 (7 files, 246.2 kB gzipped) adds `onormpvqn.cubin` and replaces 6 files of layer 16:
`KERNELS.md`, `batchfwd.py`, `onormpv.cu`, `onq8.py`, `q8.py`, `q8derive.py`. The only replaced source is `onormpv.cu`:
layer 16's copy (`7ed852e32cf86d4f…`) lacks the 11 lines layer 17's adds (no line changed or removed), and every binary
built from it (`onormpv.cubin`, `onormpvq.cubin`, `onormpvqn.cubin`) rebuilds byte for byte from this folder's
`onormpv.cu`. No binary is shadowed.

* layer 16: `sha256:585c22a8c1b0917c0a66de67685668fa58d4d8b817d614385e522f7213aa9d5f`
* layer 17: `sha256:f92353e95de6dd7356928de1873df11bd900cc0859d289dbe6e5bab953907ad5`

| binary (layer) | sha256 | source in this folder (sha256) | build line | toolkit | author (from KERNELS.md) | new since UID 220 |
|---|---|---|---|---|---|---|
| `ngemm5.cubin` (16) | `7e7775b0f52bf73c5848d5d0016070663ec1c0581389b48d06bb2a4d29585b94` | `ngemm5.cu` (`1d6410833144ece2…`) | `nvcc -arch=sm_120a -O3 -cubin -o ngemm5.cubin ngemm5.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 220) |
| `ngemm9b.cubin` (16) | `f097e26cfa1064f3e784ce0bb6bf79263491761933f4cd9d9f1ec9ef67545937` | `ngemm9b.cu` (`535c9b97c52d0e9d…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9b.cubin ngemm9b.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 220) |
| `ngemm9b_wn.cubin` (16) | `a500d86659c59a9eb0928b82bac78073a2b4b5d0b5cc165689d12cef72914373` | `ngemm9b.cu` (`535c9b97c52d0e9d…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngemm9b_wn.cubin ngemm9b.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 220) |
| `ngemm9b_wnt.cubin` (16) | `5c33f79864181ea19d529e70d533303840831452c806c8b1907fa8f836e6a773` | `ngemm9b.cu` (`535c9b97c52d0e9d…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngemm9b_wnt.cubin ngemm9b.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 220) |
| `ngemm9t.cubin` (16) | `111b70b04ac63f4c12f82145d45a1ae64bafb3c5f812fd3dae4b195b08e7f70d` | `ngemm9t.cu` (`4af699b4b0b4ad0b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9t.cubin ngemm9t.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 220) |
| `ngemm9t_wn.cubin` (16) | `daf373f578a3dfe87ab9bad1d4673ddb6af9bc2bb0e1c99593f284e23a07955b` | `ngemm9t.cu` (`4af699b4b0b4ad0b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngemm9t_wn.cubin ngemm9t.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 220) |
| `ngemm9t_wnt.cubin` (16) | `a5e3562bbd7ed512d55f51aaf872f4a7a0b3523a578845b888dc8882f79be142` | `ngemm9t.cu` (`4af699b4b0b4ad0b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngemm9t_wnt.cubin ngemm9t.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 220) |
| `ngemm9l.cubin` (16) | `f35d8d2a2d44d01a0de8154adeaac94651035aaf83aaf04e021f048d37b57e88` | `ngemm9l.cu` (`27a5d2f0eed18ec7…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngemm9l.cubin ngemm9l.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 220) |
| `ngr.cubin` (16) | `15dc8936691b884e1364a5c57f64470e354bc59c6c64c1244307fcf614f8e889` | `ngr.cu` (`7831294a4fa20335…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ngr.cubin ngr.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 220) |
| `ngr_wn.cubin` (16) | `0ecbf8d81df54457d451d357a03f1ef696ef869755c3d402b749482a10ee3fd1` | `ngr.cu` (`7831294a4fa20335…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -o ngr_wn.cubin ngr.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX-R, native rewrite 10-24: EPX's tiled-weight addresses + n-major issue order on RARN's GEMMs) | no (published with UID 220) |
| `xa.cubin` (16) | `8d119f6b271de9ccf86d58b99b5a37299cc9460b1c247f4a74e70bd8154762aa` | `xa.cu` (`231ea2f7bfd427b1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o xa.cubin xa.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 220) |
| `kkt2n.cubin` (16) | `fd5b50c85669d3f106d3415523b62651bf5e6ace69cb0531404059ba87000c3a` | `kkt2n.cu` (`05f7c62710f2e8ba…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o kkt2n.cubin kkt2n.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 220) |
| `ffs.cubin` (16) | `3c739cd2bfef61f8199601b4122dd4cda971c90015a4165284c696dea250c8f2` | `ffs.cu` (`724abb42dcca68d9…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o ffs.cubin ffs.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 220) |
| `pron.cubin` (16) | `f34ec4fb457a129b920928ea3355e689cd739a77d94b75e6236da5e053a34429` | `pron.cu` (`8bfa8dd5f059d4f9…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DPRON_SHIP -o pron.cubin pron.cu` | CUDA 13.4 (`V13.4.92`) | rgh | no (published with UID 220) |
| `onorm.cubin` (16) | `e1188ab018788db45f5630f2c8fb7dee62c120a86f8aea072846efecedf86b8c` | `onorm.cu` (`ec42b12a08a1278d…`) | `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onorm.cubin onorm.cu` | CUDA 13.4 (`V13.4.92`) | rgh (source onorm_v17.cu) | no (published with UID 220) |
| `kf.cubin` (16) | `63117c351394d8e3bf6635bde4fffbe9e5e6aff03d2aa682eeee05b102608a0c` | `kf.cu` (`ea0ebf9ab3bb6da1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o kf.cubin kf.cu` | CUDA 13.4 (`V13.4.92`) | ruj (the 'KF_NO_KERNELS' guard and the 'late' hook in 'kf.cu': kai) | no (published with UID 220) |
| `kf_wn.cubin` (16) | `da54a1b42746ad55740516daf39f66a6a8a884442a65a6ea4efbb156977d066a` | `kf.cu` (`ea0ebf9ab3bb6da1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DKF_WTL -DKF_NMAJ -o kf_wn.cubin kf.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 220) |
| `kf_wnt.cubin` (16) | `37439958d3092a8dbd14d5289e395fd85f5476c326e91952345e2b8133b2a700` | `kf.cu` (`ea0ebf9ab3bb6da1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DKF_WTL -DKF_NMAJ -DTROW -o kf_wnt.cubin kf.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX, native rewrite 10-24: WTL tiled-weight addresses + NMAJ n-major issue order, after pjs's WTL / NMAJ for NG7) | no (published with UID 220) |
| `skx6.cubin` (16) | `d5e3b11dbf53e0b5fa08bbd633d685f64e63738fdb00cd7777dd54176b83caf5` | `skx6.cu` (`4f0e74d32b6ebabf…`) | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6.cubin skx6.cu` | CUDA 13.4 (`V13.4.92`) | ruj | no (published with UID 220) |
| `skx6t.cubin` (16) | `017ca4f7f600affcd2cfb3659c98ccb26ca04e3f65993f491e57dc9cd349ca75` | `skx6t.cu` (`d64594a7372f2c39…`) | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o skx6t.cubin skx6t.cu` | CUDA 13.4 (`V13.4.92`) | ruj | no (published with UID 220) |
| `rn.cubin` (16) | `88247357dccfa88bd1d7c10eac47e3eb7e079e10498b30daacc3c5987b467487` | `rn.cu` (`8c00eac6f0c4170d…`) | `nvcc -cubin -arch=sm_120a -O3 -fmad=false -Xptxas -v -o rn.cubin rn.cu` | CUDA 13.4 (`V13.4.92`) | ruj | no (published with UID 220) |
| `ngemm7.cubin` (16) | `8671e9a9bec1573a9c5ffe94d4ace7475f38509a77e9cec346ccee79246a827f` | `ngemm7.cu` (`96ff118ce403cbab…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -o ngemm7.cubin ngemm7.cu` | CUDA 13.4 (`V13.4.92`) | pjs (nsw7l, ported by rgh) | no (published with UID 220) |
| `kfx.cubin` (16) | `8ed921f761f1632bf08a4637b4c018fe059964cf468e8303d1899dec41c992fb` | `kfx.cu` (`42529f1d88bcccb5…`) | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -o kfx.cubin kfx.cu` | CUDA 13.4 (`V13.4.92`) | kai (instantiates ruj's KF tile; includes 'kf.cu' and 'kfx_map.h') | no (published with UID 220) |
| `zcin.cubin` (16) | `f2ea51229214c944e873777b2f34812b3a6ef343686674564927331a98793133` | `zcin.cu` (`3e94fa1acd1e5695…`) | `nvcc -cubin -arch=sm_120a -O3 -Xptxas -v -o zcin.cubin zcin.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 220) |
| `xa3.cubin` (16) | `31cd78d9e28dc0ced0ae3f8283fe3ad2d8604bd5b5af159f220b9e903c612032` | `xa3.cu` (`bf836985bd3af8c7…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o xa3.cubin xa3.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 220) |
| `onormpv.cubin` (16) | `b4a29ab178559c8b8a9f8146e5ecfdc45dee9e6383dc0e6d2375f58d6e32e7b0` | `onormpv.cu` (`d9349319be794952…`) | `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -o onormpv.cubin onormpv.cu` | CUDA 13.4 (`V13.4.92`) | ruj (PROFUSE-v on ONORM; includes rgh's 'onorm.cu' unchanged; the v staging after rgh's dev_profuse ffsp.cu) | no (published with UID 220) |
| `ngemm9b_trow.cubin` (16) | `08c60e917d5bbf31d77db1ea42fadf27b2ae25b303f2ea9ddfd2bffff3413e7b` | `ngemm9b.cu` (`535c9b97c52d0e9d…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngemm9b_trow.cubin ngemm9b.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 220) |
| `ngemm9t_trow.cubin` (16) | `789b913ddf4ed3eefacfaaafefb47fd284333ca43c1110852d5609939971ad4c` | `ngemm9t.cu` (`4af699b4b0b4ad0b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngemm9t_trow.cubin ngemm9t.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 220) |
| `ngemm7_trow.cubin` (16) | `a444dd2d2b11181c283a67baeaffe14901227438734067401b3c71ae090404ee` | `ngemm7.cu` (`96ff118ce403cbab…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -DTROW -o ngemm7_trow.cubin ngemm7.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 220) |
| `ngr_trow.cubin` (16) | `6216520edc39d60ea9c5c4d1355585fac30202137e132a4b2349802dc787be09` | `ngr.cu` (`7831294a4fa20335…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o ngr_trow.cubin ngr.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 220) |
| `ngr_wnt.cubin` (16) | `913ca3a1df92bcf81599025e01377a153ce35180b3af17ce746ff705690ec113` | `ngr.cu` (`7831294a4fa20335…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DNG9_WTL -DNG9_NMAJ -DTROW -o ngr_wnt.cubin ngr.cu` | CUDA 13.4 (`V13.4.92`) | rgh (EPX-R, native rewrite 10-24: EPX's tiled-weight addresses + n-major issue order on RARN's GEMMs) | no (published with UID 220) |
| `kf_trow.cubin` (16) | `00e5eacf188344da970d8c211eb027bf3200edaeec0c1a5d5ab6bda6a94e8906` | `kf.cu` (`ea0ebf9ab3bb6da1…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -DTROW -o kf_trow.cubin kf.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 220) |
| `kfx_trow.cubin` (16) | `608e98e16f2077ef5ae143c8af69093266c8b21e50ace28572efe9c1aa17fa68` | `kfx.cu` (`42529f1d88bcccb5…`) | `nvcc -arch=sm_120a -O3 -cubin -Xptxas -v -DTROW -o kfx_trow.cubin kfx.cu` | CUDA 13.4 (`V13.4.92`) | kai | no (published with UID 220) |
| `ngemm8.cubin` (16) | `b560f9354f31354d33ad3227082b2b51e3ec235d917b78fc392cc23652519c1e` | `ngemm8.cu` (`fcbd047c076899b2…`) | `nvcc -O3 -arch=sm_120a -cubin -o ngemm8.cubin ngemm8.cu` | CUDA 13.4 (`V13.4.92`) | pjs (NG8, e17 / e20 / e22p: NG7's mainloop / epilogue / arithmetic with 192 x 192 body tiles; its t160 / t224 bands follow rgh's NG9T idea) | no (published with UID 220) |
| `ngemm7n.cubin` (16) | `7fdb6980d653f6e4425cf51ee09d14aca0c1d7e63fd53874199a318b20ff88ed` | `ngemm7n.cu` (`4ceda9b976285ecd…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -o ngemm7n.cubin ngemm7n.cu` | CUDA 13.4 (`V13.4.92`) | pjs (e27 NMAJ: ngemm7.cu with an n-major HMMA issue order, kernels '*_n') | no (published with UID 220) |
| `ngemm7w.cubin` (16) | `c3a4f7d88a1247519b508d5f1a3f78dce78bf317d72a2126de11a60678b3a34d` | `ngemm7w.cu` (`373e8af2d43090a8…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -DNG7_WTL -o ngemm7w.cubin ngemm7w.cu` | CUDA 13.4 (`V13.4.92`) | pjs (WTL twins of the nsw7l kernels incl. rgh's four NG7T tall-tail instantiations) | no (published with UID 220) |
| `ngemm7wn.cubin` (16) | `c2b076f768537061397cf6f401a02e66df67f4a42eef0a168eab5c68c1a14610` | `ngemm7wn.cu` (`da669583828fea3d…`) | `nvcc -cubin -O3 -std=c++17 -arch=sm_120a -DNG7_WTL -o ngemm7wn.cubin ngemm7wn.cu` | CUDA 13.4 (`V13.4.92`) | pjs (e27-BUILD WN: ngemm7w.cu with NMAJ's n-major k16 issue order, kernels '*_wn'; incl. rgh's four NG7T tall-tail instantiations) | no (published with UID 220) |
| `q8.cubin` (16) | `917aab0ee17f08dc16b57f81e1e721bae7c4b0480d7efb8d6fe1a723feff45f7` | `q8.cu` (`f7dcf568658de60b…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8.cubin q8.cu` | CUDA 13.4 (`V13.4.92`) | rgh (Q8, native rewrite 10-24: W8A8 int8 projections, fine-grained down / out groups, fused SwiGLU -> int8 and add + RMSNorm -> int8; ROT: 128-point Hadamard rotation of the fine groups, q8gr_* / q8x_prepfr) | no (published with UID 220) |
| `q8k.cubin` (16) | `e022f80b3fbb9c204a8399199ebbcc3f0f677de41a0a16cc07fdbe2ec3b86e64` | `q8k.cu` (`30d05fbe9f38683c…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8k.cubin q8k.cu` | CUDA 13.4 (`V13.4.92`) | ruj (r8q2 Q8K twins of rgh's Q8 kernels, ported onto g9r6 with twins of rgh's ROT kernels q8gr_* (g9x); 'q8k.cu' = '#define Q8K 1' + '#include "q8.cu"': the 'Q8K' block of 'q8.cu') | no (published with UID 220) |
| `q8pb.cubin` (16) | `365f36b43ce62126f2ce83df641d95fff634099931461f1d08a56643299d9d6e` | `q8pb.cu` (`966066811a41a83e…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8pb.cubin q8pb.cu` | CUDA 13.4 (`V13.4.92`) | pjs (Q8-PB / Q8-NX, e35: twins of rgh's q8f_ fine-group kernels whose epilogue stores bf16 rows - _pbd: bf16x2 register stores, _pb: rgh's store_tile - instead of the fp32 partial, and rgh's q8n_addnorm with a null-x option (q8n_addnorm_nx); q8pb.cu = rgh's j206 q8.cu namespace + tile_fpb / hetero_fpb / store_rows_bf16 + q8n_addnorm_nx; e36: + Q8-S96, the q8f twins with a 96-row tail tile (*_s96x128_*)) | yes (changed; UID 220's is `9d85f48101785b51…`) |
| `q8f2.cubin` (16) | `b93b8377031b7ab3b5e0c7e64fa6a749711e976fa5124ec47f1fd75f97ec7d4f` | `q8f2.cu` (`b95b6766eca41cdb…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8f2.cubin q8f2.cu` | CUDA 13.4 (`V13.4.92`) | rgh (Q8F2 10-25: the fine int8 projections (down_proj, GDN out_proj, attention o_proj) with 64- and 96-row big tiles - pjs's Q8-PB tile_fpb (bf16x2 register stores) with rgh's mainloop_fine on rgh's s64x128 / pjs's s96x128 tile configurations, byte-identical to q8f_*_pbd at every R; kernels q8f2_b64x128_s{32,64}x128, q8f2_b96x128_s{32,64,96}x128) | yes (new file) |
| `q8ri.cubin` (16) | `0e7b847d740882738a5867556fd7f284aacf826e43ac75bcc8428e1f6a82d9c0` | `q8ri.cu` (`704420ccd2e345d7…`) | `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o q8ri.cubin q8ri.cu` | CUDA 13.4 (`V13.4.92`) | rgh (RIN 10-25: the in families' int8 inputs rotated per 128-channel group before the row scale; q8n_addnormr / q8x_prepr) | yes (new file) |
| `q8rf.cubin` (16) | `8eeb143b2c7993a2d352b30d715b027d2cb220b52211f0786098d74f352d21dc` | `q8rf.cu` (`03a33d70078327b5…`) | `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o q8rf.cubin q8rf.cu` | CUDA 13.4 (`V13.4.92`) | rgh (ROTF 10-25: the in families' int8 inputs rotated over the full 2560-channel row - RIN's per-group R_128, then the Paley H_20 group mix; q8n_addnormf / q8x_prepf) | yes (new file) |
| `onormpvq.cubin` (16) | `da628c4f393148b8ab1342be49cfceffcb1f83c3bfe3d8efba618fdcd93c2590` | `onormpv.cu` (`d9349319be794952…`) | `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -DONORM_Q8 -o onormpvq.cubin onormpv.cu` | CUDA 13.4 (`V13.4.92`) | rgh (ONQ8, 10-25: ruj's PROFUSE-v kernel whose gated-norm epilogue also writes the GDN out_proj's int8 input + group scales, q8x_prepfr's bytes; the '#ifdef ONORM_Q8' blocks of 'onormpv.cu') | yes (new file) |
| `onormpvqn.cubin` (17) | `d2f8eb264e01e191318735028b135be1fde3b5dec7a900c7f0b9ef719c189104` | `onormpv.cu` (`d9349319be794952…`) | `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -DFFS_SHIP -DONORM_V=5 -DONORM_PF2 -DONORM_SILU_FIN -DONORM_FINROLL -DONORM_Q8 -DONORM_Q8_NOO -o onormpvqn.cubin onormpv.cu` | CUDA 13.4 (`V13.4.92`) | ruj (ONQ8N, r5cx 10-25: rgh's ONQ8 kernel 'ffs_pv5nq' without the global store of the bf16 gated-norm rows - XQP's idea on rgh's kernel; the '#ifdef ONORM_Q8_NOO' blocks of 'onormpv.cu') | yes (new file) |
| `q8gq.cubin` (16) | `86a50629877c834067a45b409e56a805a50312e400450f98ca3f900f3c75915d` | `q8gq.cu` (`5964486500998fa0…`) | `nvcc -O3 -arch=sm_120a -cubin -std=c++17 -Xptxas -v -o q8gq.cubin q8gq.cu` | CUDA 13.4 (`V13.4.92`) | rgh (DERIVE 10-25: GPTQ's column loop of one 128-column block in one launch, one warp per row; the torch loop's IEEE ops bit for bit) | yes (new file) |
| `q8h2.cubin` (16) | `9d869805a33492797f98dbd1c187db5b4911a35edccabe2ba145620cbfc38746` | `q8h2.cu` (`06a498684f21c2ac…`) | `nvcc -O3 -arch=sm_120a -cubin -Xptxas -v -o q8h2.cubin q8h2.cu` | CUDA 13.4 (`V13.4.92`) | rgh (Q8H2 10-25: persistent byte-identical twins of the in-family Q8 kernels at H = 0 - q8h2_h_* for q8h x256 (qkv | yes (new file) |
| `ftokc.so` (16) | `2b34d5b4b7b56dfbe38fa57be16ac1005da54b1b64ce886cdb02bc0ea0e224dc` | `ftokc.c` (`1fe352dc6cf51083…`) | `gcc -O2 -fPIC -shared -nostdlib -ffreestanding -fno-builtin -o ftokc.so ftokc.c` | gcc 13.3.0 (Ubuntu 24.04 package `13.3.0-6ubuntu2~24.04.1`) | kai | no (published with UID 220) |

Nothing else in layers 16-17 is compiled (all 48 ELF files in them are the 48 binaries above, all in
`/opt/s1eng/s1eng`). The other files are the CUDA / C sources above, Python source, JSON tables, notes (`KERNELS.md`)
and, under `/opt/s1eng/levers`, the lever notes, patches, env files, harness / test scripts and copies of `skx6.cu` /
`skx6t.cu` (byte-identical to the shipped sources). There is no weight or calibration file: the Q8 weights are derived
at start-up.

## Notes

* `-Xptxas -v` matters where a build line has it: ptxas writes its own command line into the cubin's `.note.nv.tkinfo`.
* `onorm.cubin`, `onormpv.cubin`, `onormpvq.cubin`, `onormpvqn.cubin`: `onorm.cu` (also included by `onormpv.cu`) has a
  device function in an anonymous namespace, whose internal name includes the CRC32 of the source's absolute path, so
  they are byte-identical only when built from `/opt/s1eng/s1eng/`. Built elsewhere, only the string table and the
  symbol tables that point into it differ.
* `ftokc.so` is freestanding (no libc, no NEEDED entry) and built with Ubuntu 24.04's gcc 13.3.0.
* Includes: `kfx.cu` includes `kf.cu` and `kfx_map.h`; `onormpv.cu` includes `onorm.cu`; `q8f2.cu` includes `q8.cu`;
  `q8h2.cu` includes `q8.cu`; `q8k.cu` includes `q8.cu`; `q8rf.cu` includes `q8ri.cu`.

## How to check

```
./verify.sh <path to nvcc 13.4>
```

`verify.sh` reads the committed image's manifest by digest (`crane manifest`) and checks its sha256 and the pinned layer
digests in it, downloads layers 16 and 17 (`crane blob`), checks each blob's sha256 and unpacks it, rebuilds every
binary from the sources here and compares the sha256 with the layer's copy (48 copies of the 48 binaries), and checks
that every ELF file in those layers is one of the 48. It reads the image anonymously from ghcr.io. It needs `crane`,
`tar`, `sha256sum`, `od` and `docker`, and builds the four `onorm` cubins at `/opt/s1eng/s1eng` when that directory can
be created (run it as root, with no `onorm.cu` there); without it they are built in the scratch folder, report DIFFERS
(only the string and symbol tables differ) and the script exits 1. `ftokc.so` is built inside `ubuntu:24.04` (pinned by
digest) with the gcc-13 package `13.3.0-6ubuntu2~24.04.1`. The script exits 1 when the manifest or a layer does not
check out, a binary is found in no layer, or a layer holds a compiled file that is not rebuilt here.

Local test mode (no registry): on a machine whose docker uses the containerd image store and already holds the image,
put this stand-in for `crane` first on `PATH` and run `verify.sh` as root; it serves the manifest and the blobs from
docker's content store, and the manifest / blob sha256 checks apply unchanged:

```sh
#!/bin/sh
# crane stand-in (local test mode): "crane manifest|blob <repo>@<digest>" read from the local docker's content store
case $1 in manifest|blob) ;; *) echo "local test mode: only crane manifest|blob <repo>@<digest>" >&2; exit 2;; esac
exec ctr -n moby content get "${2#*@}"
```

## Authors

The engine is the team's work: rgh, ruj, kai and pjs (per binary in the table above).

## Licence

See `LICENSE` (MIT).
