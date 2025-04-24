#!/usr/bin/env python
import glob
import numpy as np
import os
import SimpleITK as sitk
import sys
import yn_tools.kinetic as ykin
import yn_tools.image_manip as yimg

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

frame_timing = [(6, 30), (3, 60), (2, 120), (22, 300)]
frame_durations = ykin.frame_timing_to_frame_durations(frame_timing)
frame_ranges = ykin.frame_durations_to_frame_ranges(frame_durations, scale=1000)
frame_ranges = [(int(f[0]), int(f[1])) for f in frame_ranges]  # Make sure it's integers
num_frames = len(frame_ranges)

# %% Get decay factors



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
num_rois = len(roi_names)

roi_images = list()
roi_images_np = list()
roi_paths = list()

for roi_i in range(num_rois):
    roi_name = roi_names[roi_i]
    roi_path_glob_fname = os.path.join(
        roi_dir, f"mrtac_qcfile_{roi_name}_nonlin__*_weights.nii.gz"
    )
    roi_path_glob = glob.glob(roi_path_glob_fname)
    if len(roi_path_glob) > 1:
        sys.exit(f"There is more than one ROI file in the folder for {roi_name}")
    if len(roi_path_glob) < 1:
        sys.exit(f"There is no ROI file in the folder for {roi_name}")
    roi_path = roi_path_glob[0]
    roi_paths.append(roi_path)
    roi_image = sitk.ReadImage(roi_path)
    roi_images.append(roi_image)
    roi_images_np.append(sitk.GetArrayViewFromImage(roi_image))

# %% Read the PET images

yrtpet_recon_images = list()  # One SITK image here per frame
yrtpet_recon_images_np = list()  # One SITK image here per frame

for frame_i in range(num_frames):
    yrtpet_recon_image_path = os.path.join(
        yrtpet_recon_dir, f"frame_{str(frame_i).zfill(2)}", "recon_image.nii.gz"
    )
    yrtpet_recon_image = sitk.ReadImage(yrtpet_recon_image_path)
    yrtpet_recon_images.append(yrtpet_recon_image)
    yrtpet_recon_images_np.append(sitk.GetArrayViewFromImage(yrtpet_recon_image))

# %% Resample the masks in the PET images

pet_reference_image = yrtpet_recon_images[0]  # For resampling only
roi_full_image_np = np.zeros((990, 600, 600), dtype=np.float32)

roi_images_resampled = list()
roi_images_resampled_np = list()

for i in range(num_rois):
    roi_name = roi_names[i]

    roi_image_resampled_path = os.path.join(
        out_dir, f"roi_{roi_name}_weights_resampled.nii.gz"
    )

    roi_path = roi_paths[i]
    roi_image_np = roi_images_np[i]

    if not os.path.isfile(roi_image_resampled_path):
        roi_full_image_np[:] = 0
        roi_full_image_np[732:914, 150:304, 72:216] = roi_image_np
        roi_full_image_np[:] = roi_full_image_np[::-1, ::-1, :]

        roi_full_image = sitk.GetImageFromArray(roi_full_image_np)
        roi_full_image.SetSpacing((0.5, 0.5, 0.5))
        roi_full_image.SetOrigin(
            [
                -vd * (nd - 1) / 2
                for (nd, vd) in zip(
                    roi_full_image.GetSize(), roi_full_image.GetSpacing()
                )
            ]
        )
        roi_image_resampled = yimg.resample_to_reference(
            roi_full_image, pet_reference_image
        )
        sitk.WriteImage(roi_image_resampled, roi_image_resampled_path)

        roi_images_resampled.append(roi_image_resampled)
        roi_image_resampled_np = sitk.GetArrayViewFromImage(roi_image_resampled)
        roi_images_resampled_np.append(roi_image_resampled)
    else:
        roi_image_resampled = sitk.ReadImage(roi_image_resampled_path)
        roi_images_resampled.append(roi_image_resampled)
        roi_image_resampled_np = sitk.GetArrayViewFromImage(roi_image_resampled)
        roi_images_resampled_np.append(roi_image_resampled_np)
        pass


# TODO: %% Generate Time-Activity-Curves of the PET images

yrtpet_roi_tacs = [np.zeros((num_frames))] * num_rois

for roi_i in range(num_rois):
    roi_name = roi_names[roi_i]
    print(f"ROI {roi_name}")

    yrtpet_roi_tac_path = os.path.join(out_dir, f"tac_{roi_name}_yrtpet.npy")

    if not os.path.isfile(yrtpet_roi_tac_path):
        roi_image_resampled_np = roi_images_resampled_np[roi_i]
        
        for frame_i in range(num_frames):
            
            yrtpet_avg_in_roi = (
                yrtpet_recon_images_np[frame_i] * roi_image_resampled_np
            ).sum() / roi_image_resampled_np.sum()
            
            yrtpet_roi_tacs[roi_i][frame_i] = yrtpet_avg_in_roi

        np.save(yrtpet_roi_tac_path, yrtpet_roi_tacs[roi_i])
    else:
        yrtpet_roi_tacs[roi_i] = np.load(yrtpet_roi_tac_path)
