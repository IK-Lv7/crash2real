# Third-party code, data and models

None of these are included in this repository; they are downloaded and used. Check each source for the current terms (below is what we read when we checked).

| Name | Used for | License (as we read it) |
| --- | --- | --- |
| [drivestudio](https://github.com/ziyc/drivestudio) (commit e59bda4) | 3DGS reconstruction and rendering; small changes in `patches/` | see the repository |
| [Argoverse 2 Sensor](https://www.argoverse.org/av2.html) | driving logs (images, LiDAR, maps, annotations) | CC BY-NC-SA 4.0 (non-commercial, share alike) |
| [SegFormer](https://github.com/NVlabs/SegFormer) | sky masks during preprocessing | see the repository |
| [pymunk](https://www.pymunk.org/) | 2D rigid-body simulation after the impact | MIT |
| [Difix (nvidia/difix)](https://github.com/nv-tlabs/Difix3D) | optional refinement of rendered frames | NVIDIA License (non-commercial only) |

The crash-test numbers in `validation/crash_tests.py` were read from public reports and figures (NHTSA test reports v08549R003 and v03949R001; Gidlewski, Prochowski & Wach 2019; RICSAC test 1 as tabulated in the Virtual CRASH guide). The documents are not included; sources are listed in the script.
