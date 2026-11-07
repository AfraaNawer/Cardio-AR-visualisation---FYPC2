# G24 – Augmented Reality System for Multimodal Cardiovascular Visualisation

ENG40008 Final Year Project, Swinburne University of Technology, Semester 2 2026.
Afra Nawer and Allqa Ashna Ali. Supervisor: Prof. Paul Stoddart.

## Contents
| Folder | Contents |
|---|---|
| `slicer/` | `slicer_pipeline.py` – run in the 3D Slicer Python console. Builds 7 CT + 7 MRI structures, aligns MRI to CT, prints the residuals and exports 14 STL files. |
| `pi/` | `heart_viewer.py` (interactive 3D viewer), `ar_body.py` (markerless AR), `ar_heart.py` (marker-based AR prototype). |
| `tests/` | Test data from the final report (T2–T6) and the chart scripts. |
| `cad/` | Case v2: parametric model (`pi5_case_v2.py`, manifold3d), STL files for printing and STEP files for CAD. |

Not included: the MM-WHS images (download from the MM-WHS challenge), the heart meshes (recreated by `slicer_pipeline.py`) and
`pose_landmarker_lite.task` (download from Google MediaPipe and place in `pi/`).

## Quick start (Raspberry Pi 5, Raspberry Pi OS 64-bit, desktop session)
```
python3 heart_viewer.py export/ct export/mr                 # 3D viewer
~/mpenv/bin/python ar_body.py export/ct export/mr           # AR mode (MediaPipe installed in a virtual environment)
```
Test options: `--bench <seconds>` `--log <file.csv>` (`--uncapped` for the viewer, `--poses 2` for AR).
Viewer keys: drag rotate · scroll zoom · 1–9 structures · F1–F4 layers · T see-through · A auto-rotate · R reset · S screenshot · Q quit.
AR keys: 1–9 · C/M layer · T see-through · G ghost · +/− size · arrows turn/tilt · K key points and shoulder line · S save picture · R reset · Q quit.

Fit the Raspberry Pi 5 active cooler before long use: without it the processor reached 85 °C during testing.

This is a research and teaching prototype. It is not a medical device.
