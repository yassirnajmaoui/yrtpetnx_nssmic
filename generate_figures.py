#!/usr/bin/env python
import SimpleITK as sitk
import numpy as np
import os
import yn_tools.display as ydisp
import matplotlib.pyplot as plt

# %% ======================== Mini-Derenzo section ========================

# TODO: Generate line plots for Derenzo phantom to compare YRT-PET to URT and MOLAR

#iteration_range = range(0,7) # Full, for PSNR/iteration calculation
iteration_range = range(6,7)

# %% Read YRT-PET's images

yrtpet_recon_dir = (
    "/data24/yn257/nx_derenzo/recons/watt_wnorm_siddon_wdoi_wtof_wpsf_7i10s_wtofdoifix"
)
yrtpet_recon_paths = list()

for iteration in iteration_range:
    if iteration+1 == 7:
        yrtpet_recon_paths.append(os.path.join(yrtpet_recon_dir, "recon_image.nii.gz"))
    else:
        yrtpet_recon_paths.append(
            os.path.join(yrtpet_recon_dir, f"recon_image_iteration{iteration+1}.nii.gz")
        )

yrtpet_recon_images = list()
yrtpet_recon_images_np = list()
for i in range(len(yrtpet_recon_paths)):
    yrtpet_recon_images.append(sitk.ReadImage(yrtpet_recon_paths[i]))
    yrtpet_recon_images_np.append(sitk.GetArrayViewFromImage(yrtpet_recon_images[-1]))

# %% Read MOLAR's images

molar_recon_dir = (
    "/data2/Recons/kf283/nx_paper/20230728_MD000/injid_33899/listmode_001/wDOI/Frame037"
)
molar_recon_paths = list()
molar_recon_images = list()
molar_recon_images_np = list()
for iteration in iteration_range:
    molar_recon_path = os.path.join(
        molar_recon_dir, f"20230728_md000_F-18Solution-{iteration+1}.img"
    )
    molar_recon_paths.append(molar_recon_path)
    molar_recon_images.append(sitk.ReadImage(molar_recon_path))
    molar_recon_images_np.append(sitk.GetArrayViewFromImage(molar_recon_images[-1]))

fig_lineplot = ydisp.matshow_images_with_lineplot(
    [molar_recon_images[-1], yrtpet_recon_images[-1]],
    zslice=474,
    x1=260,
    y1=414,
    x2=250,
    y2=363,
    xlim=(235, 395),
    ylim=(297, 467),
    labels=["MOLAR", "YRT-PET"],
    normalize=True,
    vmaxs=[0.7935e6, 1.5],
    fig_height=4,
    margin_bottom=0.07
)

plt.show()
