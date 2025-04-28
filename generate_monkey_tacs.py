#!/usr/bin/env python
import glob
import numpy as np
import os
import SimpleITK as sitk
import yn_tools.kinetic as ykin
import yn_tools.radioactive as yrad
import yn_tools.image_manip as yimg

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
num_rois = len(roi_names)

roi_images = list()
roi_images_np = list()
roi_paths = list()

for roi_i in range(num_rois):
    roi_name = roi_names[roi_i]
    roi_path_glob_path = os.path.join(
        roi_dir, f"mrtac_qcfile_{roi_name}_nonlin__*_weights.nii.gz"
    )
    roi_path_glob = glob.glob(roi_path_glob_path)
    if len(roi_path_glob) > 1:
        raise RuntimeError(
            f"There is more than one ROI file in the folder for {roi_name}"
        )
    if len(roi_path_glob) < 1:
        raise RuntimeError(f"There is no ROI file in the folder for {roi_name}")
    roi_path = roi_path_glob[0]
    roi_paths.append(roi_path)
    roi_image = sitk.ReadImage(roi_path)
    roi_images.append(roi_image)
    roi_images_np.append(sitk.GetArrayViewFromImage(roi_image))

# %% Read the PET images from YRT-PET

yrtpet_recon_images = list()  # One SITK image per frame
yrtpet_recon_images_np_path = os.path.join(out_dir, "yrtpet_recons.npz")

if not os.path.isfile(yrtpet_recon_images_np_path):
    yrtpet_recon_images_np = np.zeros((num_frames, *image_dimensions))

    for frame_i in range(num_frames):
        print(f"Reading YRT-PET image for frame {frame_i}")
        yrtpet_recon_image_path = os.path.join(
            yrtpet_recon_dir, f"frame_{str(frame_i).zfill(2)}", "recon_image.nii.gz"
        )
        yrtpet_recon_image = sitk.ReadImage(yrtpet_recon_image_path)
        yrtpet_recon_images.append(yrtpet_recon_image)
        yrtpet_recon_image_np = sitk.GetArrayFromImage(yrtpet_recon_image)

        # Apply scaling on the NumPy array
        yrtpet_recon_image_np *= (
            1
            / (molar_global * frame_durations[frame_i])
            / decay_factors[frame_i]
            / livetime_factors[frame_i]
        )

        yrtpet_recon_images_np[frame_i] = yrtpet_recon_image_np

    np.savez_compressed(yrtpet_recon_images_np_path, yrtpet_recon_images_np)
else:
    yrtpet_recon_images_np = np.load(yrtpet_recon_images_np_path)

# %% Read the PET images from MOLAR

molar_recon_images_np_path = os.path.join(out_dir, "molar_recons.npz")

if not os.path.isfile(molar_recon_images_np_path):

    molar_recon_images_np = np.zeros((num_frames, *image_dimensions))

    for frame_i in range(num_frames):
        print(f"Reading MOLAR image for frame {frame_i}")
        molar_recon_image_glob_path = os.path.join(
            molar_recon_dir, f"Frame{str(frame_i+1).zfill(3)}", "*-7.img"
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

        molar_recon_images_np[frame_i] = np.fromfile(
            molar_recon_image_path, dtype=np.float32
        ).reshape(image_dimensions)

    np.savez_compressed(molar_recon_images_np_path, molar_recon_images_np)
else:
    molar_recon_images_np = np.load(molar_recon_images_np_path)

# %% Resample the masks in the PET grid

pet_reference_image = yrtpet_recon_images[0]  # For resampling only
roi_full_image_np = np.zeros((990, 600, 600), dtype=np.float32)

roi_images_resampled = list()
roi_images_resampled_np = list()

for i in range(num_rois):
    roi_name = roi_names[i]
    print(f'Preparing ROI "{roi_name}"')

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


# %% Generate Time-Activity-Curves of the PET images from YRT-PET and MOLAR

yrtpet_roi_tacs = [np.zeros((num_frames))] * num_rois
molar_roi_tacs = [np.zeros((num_frames))] * num_rois

for roi_i in range(num_rois):
    roi_name = roi_names[roi_i]
    print(f"Computing/Gathering TAC for ROI {roi_name}")

    yrtpet_roi_tac_path = os.path.join(out_dir, f"tac_{roi_name}_yrtpet.npy")
    molar_roi_tac_path = os.path.join(out_dir, f"tac_{roi_name}_molar.npy")

    for roi_tac_path, recon_images_np, roi_tacs in zip(
        [yrtpet_roi_tac_path, molar_roi_tac_path],
        [yrtpet_recon_images_np, molar_recon_images_np],
        [yrtpet_roi_tacs, molar_roi_tacs],
    ):
        if not os.path.isfile(roi_tac_path):
            roi_image_resampled_np = roi_images_resampled_np[roi_i]

            for frame_i in range(num_frames):

                avg_in_roi = (
                    recon_images_np[frame_i] * roi_image_resampled_np
                ).sum() / roi_image_resampled_np.sum()

                roi_tacs[roi_i][frame_i] = avg_in_roi

            np.save(roi_tac_path, roi_tacs[roi_i])
        else:
            pass
            roi_tacs[roi_i] = np.load(roi_tac_path)
