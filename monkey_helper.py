#!/usr/bin/env python
import os
import glob
import numpy as np
import yn_tools.kinetic as ykin

# %% Constants

molar_global = 3.188e-9

image_voxel_size = (1.0, 0.8, 0.8)  # ZYX
image_dimensions = (495, 630, 630)

# %% Prepare paths

out_dir = "/data24/yn257/yrtpetnx_nssmic_sandbox/monkey"
roi_dir = "/data8/data/app311_tm_rhm/20250130_sm580/app311_brain/roi_qcs"
yrtpet_recon_dir = (
    "/data24/yn257/nx_monkey/recons/wbigvoxels_watt_wnorm_siddon_wdoi_wpsf_wtof_7i10s"
)
molar_recon_dir = (
    "/data2/Recons/tm2225/20250130_sm580_baseline/injid_38594/listmode_001/"
)

# %% Compute frame timing

frame_timing = [(6, 30), (3, 60), (2, 120), (16, 300)]
frame_durations = ykin.frame_timing_to_frame_durations(frame_timing)
frame_ranges = ykin.frame_durations_to_frame_ranges(frame_durations)
frame_ranges = [(int(f[0]), int(f[1])) for f in frame_ranges]  # Make sure it's integers
num_frames = len(frame_ranges)

# %% Get factors from MOLAR's ".fi" files

decay_factors = np.zeros((num_frames), dtype=np.float32)
livetime_factors = np.zeros((num_frames), dtype=np.float32)

for frame_i in range(num_frames):
    # Find the ".fi" file
    frame_fi_file_flob = glob.glob(
        os.path.join(molar_recon_dir, f"Frame{str(frame_i+1).zfill(3)}", "*.fi")
    )
    if len(frame_fi_file_flob) > 1:
        raise RuntimeError(
            f'There is more than one ".fi" file in the folder for frame {frame_i}'
        )
    if len(frame_fi_file_flob) < 1:
        raise RuntimeError(f'There is no ".fi" file in the folder for frame {frame_i}')
    frame_fi_file = frame_fi_file_flob[0]

    # Read the ".fi" file for the frame
    with open(frame_fi_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("# Average Decay Factor"):
                decay_factors[frame_i] = float(lines[i + 1].strip())
            if line.strip().startswith("# Average LiveTimeFactor"):
                livetime_factors[frame_i] = float(lines[i + 1].strip())

    # Sanity check
    assert decay_factors[frame_i] != 0
    assert livetime_factors[frame_i] != 0

# %% Read the region masks in roi_dir

roi_names = [
    "amygdala",
    "ant_cng_rev",
    "brainstem",
    "caudate",  # Show
    "cerebellum_gm",
    "cerebellum",
    "cerebellum_wm",
    "cingulate",
    "frontal",
    "globus_pallidus",
    "hippocampus",
    "insula",
    "nacc",
    "occipital",
    "pons",
    "putamen",  # Show
    "semiovale",  # Interesting to show
    "sub_nigra",
    "temporal",
    "thalamus",
]
roi_names_clean = [
    "Amygdala", # 0
    "ant_cng_rev", # 1
    "Brainstem", # 2
    "Caudate",  # 3
    "Cerebellum grey matter", # 4
    "Cerebellum", # 5
    "Cerebellum white matter", # 6
    "Cingulate", # 7
    "Frontal", # 8
    "Globus pallidus", # 9
    "Hippocampus", # 10
    "Insula", # 11
    "Nucleus accumbens", # 12
    "Occipital", # 13
    "Pons", # 14
    "Putamen",  # 15
    "Semiovale",  # 16
    "Sub nigra", # 17
    "Temporal", # 18
    "Thalamus", # 19
]
num_rois = len(roi_names)


def get_molar_image_path(molar_dir: str, frame: int, iteration: int = 7):
    molar_recon_image_glob_path = os.path.join(
        molar_dir, f"Frame{str(frame+1).zfill(3)}", f"*-{iteration}.img"
    )
    molar_recon_image_glob = glob.glob(molar_recon_image_glob_path)
    if len(molar_recon_image_glob) > 1:
        raise RuntimeError(
            f"There is more than one MOLAR recon file in the folder for frame {frame_i}"
        )
    if len(molar_recon_image_glob) < 1:
        raise RuntimeError(
            f"There is no MOLAR recon file in the folder for frame {frame_i}"
        )
    molar_recon_image_path = molar_recon_image_glob[0]
    return molar_recon_image_path

def get_full_yrtpet_scale(frame_i: int):
    return (
        1
        / (molar_global * frame_durations[frame_i])
        / decay_factors[frame_i]
        / livetime_factors[frame_i]
    )

