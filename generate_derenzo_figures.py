#!/usr/bin/env python
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import yn_tools.display as ydisp

# iteration_range = range(0,7) # Full, for PSNR/iteration calculation
iteration_range = range(6, 7)
scan_duration = 599951  # in ms, around 10 minutes
average_CijNorm = 0.923966

# %% Read URT's images

urt_recon_dir = "/data2/Recons/jz729/For_Kathryn/20230728_MD000_145432/recon_noRC_noSC_antiPSF_iter_ready/all_recons/t00001"
urt_recon_paths = list()
urt_recon_images = list()
urt_recon_images_np = list()
for iteration in iteration_range:
    urt_recon_path = os.path.join(urt_recon_dir, f"post_recon-{iteration+1}.nii")
    urt_recon_paths.append(urt_recon_path)
    urt_recon_image = sitk.ReadImage(urt_recon_path)
    urt_recon_image = urt_recon_image[::-1, :, ::-1]
    urt_recon_images.append(urt_recon_image)
    urt_recon_images_np.append(sitk.GetArrayViewFromImage(urt_recon_image))

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
    molar_recon_image = sitk.ReadImage(molar_recon_path)
    molar_recon_images.append(molar_recon_image)
    molar_recon_images_np.append(sitk.GetArrayViewFromImage(molar_recon_image))

# %% Read YRT-PET's images

yrtpet_recon_dir = (
    "/data24/yn257/nx_derenzo/recons/watt_wnorm_siddon_wdoi_wtof_wpsf_7i10s_wtofdoifix"
)
yrtpet_recon_paths = list()

for iteration in iteration_range:
    if iteration + 1 == 7:
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

# %% Line plot

fig_lineplot = ydisp.matshow_images_with_lineplot(
    [
        urt_recon_images[-1] / average_CijNorm,
        molar_recon_images[-1],
        yrtpet_recon_images[-1] * scan_duration * average_CijNorm,
    ],
    zslice=474,
    x1=260,
    y1=414,
    x2=250,
    y2=363,
    xlim=(235, 395),
    ylim=(297, 467),
    labels=["URT", "MOLAR", "YRT-PET"],
    image_label_size=12,
    normalize=False,
    vmaxs=[8e5, 8e5, 8e5],
    fig_height=6,
    margin_bottom=0.08,
    margin_left=0.07,
)
plt.show()
