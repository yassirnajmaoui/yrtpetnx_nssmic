#!/usr/bin/env python
import glob
import numpy as np
import os
import SimpleITK as sitk
import yn_tools.image_manip as yimg
import matplotlib.pyplot as plt
import yn_tools.display as ydisp
from monkey_helper import *

# %% Constants

plot_monkey_slice: bool = False
plot_monkey_tacs: bool = True

# %% Plot monkey frame with zoom

if plot_monkey_slice:

    frame_to_show = 9
    yslice_to_show = 350

    z_crop = (0, 410)
    x_crop = (140, 415)

    molar_recon_path = get_molar_image_path(molar_recon_dir, frame_to_show)
    molar_recon_image = sitk.ReadImage(molar_recon_path)
    molar_recon_image_np = sitk.GetArrayViewFromImage(molar_recon_image)
    molar_recon_slice_np = molar_recon_image_np[
        z_crop[0] : z_crop[1], yslice_to_show, x_crop[0] : x_crop[1]
    ]

    yrtpet_recon_path = os.path.join(
        yrtpet_recon_dir, f"frame_{str(frame_to_show).zfill(2)}", "recon_image.nii.gz"
    )
    yrtpet_recon_image = sitk.ReadImage(yrtpet_recon_path)
    yrtpet_recon_image_np = sitk.GetArrayViewFromImage(yrtpet_recon_image)
    yrtpet_scale = get_full_yrtpet_scale(frame_to_show)
    yrtpet_recon_slice_np = (
        yrtpet_recon_image_np[
            z_crop[0] : z_crop[1], yslice_to_show, x_crop[0] : x_crop[1]
        ]
        * yrtpet_scale
    )

    aspect = image_voxel_size[0] / image_voxel_size[2]  # dz/dx

    # TODO: Add color bar
    fig = ydisp.matshow_images_with_zoom(
        (molar_recon_slice_np, yrtpet_recon_slice_np),
        ("MOLAR", "YRT-PET"),
        (24, 38, 132, 130),
        aspect=aspect,
        vmaxs=(4e5, 4e5),
        margin_image_label_x=0.01,
        margin_image_label_y=0.055,
        image_label_size=12,
    )
    plt.show()

if plot_monkey_tacs:

    unit_scale = 1.0 / 1000000.0  # Turn Bq to MBq
    
    yrtpet_roi_tacs = np.zeros((num_rois, num_frames))
    molar_roi_tacs = np.zeros((num_rois, num_frames))

    for roi_i in range(num_rois):
        roi_name = roi_names[roi_i]
        print(f"Gathering TAC for ROI {roi_name}")

        yrtpet_roi_tacs[roi_i] = np.load(
            os.path.join(out_dir, f"tac_{roi_name}_yrtpet.npy")
        ) * unit_scale
        molar_roi_tacs[roi_i] = np.load(
            os.path.join(out_dir, f"tac_{roi_name}_molar.npy")
        ) * unit_scale
    
    rois_to_show = [3, 5, 15, 16]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    axs = axs.flatten()  # Flatten to iterate easily
    timeaxis = np.array(frame_ranges).mean(axis=1)
    
    ylim = (0, np.max(molar_roi_tacs)*1.05)

    for i, ax in enumerate(axs):
        roi_to_show = rois_to_show[i]
        ax.plot(timeaxis, molar_roi_tacs[roi_to_show], label='MOLAR', linestyle='-', marker='.')
        ax.plot(timeaxis, yrtpet_roi_tacs[roi_to_show], label='YRT-PET', linestyle='-', marker='.')
        ax.set_title(roi_names_clean[roi_to_show], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim((0, None))
        ax.set_ylim(ylim)

        if i == 0:
            ax.legend()
        if i == 0 or i == 2:
            ax.set_ylabel('Activity [MBq/mL]')
        if i == 2 or i == 3:
            ax.set_xlabel('Time (s)')

    plt.show()
    
