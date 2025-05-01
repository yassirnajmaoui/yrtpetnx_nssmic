#!/usr/bin/env python
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import yn_tools.display as ydisp
import yn_tools.radioactive as yrad

# %% Constants

# iteration_range = range(0,7) # Full, for PSNR/iteration calculation
iteration_range = range(6, 7)
scan_duration = 599.951  # in s, around 10 minutes
molar_global = 3.407e-9
positron_fraction = 0.967
unit_scale = 1.0 / 1000.0  # Turn Bq to kBq

figures_dir = "/home1/yn257/work/data/yrtpetnx_nssmic/figures"

plot_lineplot: bool = True
plot_convergence: bool = True

# %% Calculate decay factors

# Injection time: 14:21:00
# Scan start time: 14:59:04
elapsed_injectiontime_to_scanstart = 38 * 60 + 4
decay_factor_injectiontime_to_scanstart = yrad.decay(elapsed_injectiontime_to_scanstart)
decay_factor_injectiontime_to_scanend = yrad.decay(
    elapsed_injectiontime_to_scanstart + scan_duration
)
decay_factor_scanstart_to_scanend = yrad.decay(scan_duration)
decay_factor_within_frame = yrad.decay_within_frame(0, scan_duration)
livetime_factor = 0.957977  # From the MOLAR file

# %% Read URT's images

urt_scale = (
    decay_factor_within_frame
    / decay_factor_injectiontime_to_scanend
    / livetime_factor
    / positron_fraction
)

urt_recon_dir = "/data2/Recons/jz729/For_Kathryn/20230728_MD000_145432/recon_noRC_noSC_antiPSF_iter_ready/all_recons/t00001"
urt_recon_paths = list()
urt_recon_images = list()
urt_recon_images_np = list()
for iteration in iteration_range:
    urt_recon_path = os.path.join(urt_recon_dir, f"post_recon-{iteration+1}.nii")
    urt_recon_paths.append(urt_recon_path)
    urt_recon_image = sitk.ReadImage(urt_recon_path)
    urt_recon_image = urt_recon_image[::-1, :, ::-1] * urt_scale
    urt_recon_images.append(urt_recon_image)
    urt_recon_images_np.append(sitk.GetArrayViewFromImage(urt_recon_image))

# %% Read MOLAR's images

molar_scale = 1 / decay_factor_injectiontime_to_scanend

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
    molar_recon_image = sitk.ReadImage(molar_recon_path) * molar_scale
    molar_recon_images.append(molar_recon_image)
    molar_recon_images_np.append(sitk.GetArrayViewFromImage(molar_recon_image))

# %% Read YRT-PET's images

yrtpet_scale = (
    1 / (scan_duration * molar_global) / decay_factor_injectiontime_to_scanend
)

yrtpet_recon_dir = (
    "/data24/yn257/nx_derenzo/recons/watt_wnorm_siddon_wdoi_wtof_wpsf_7i10s_wtofdoifix"
)
yrtpet_recon_paths = list()
yrtpet_recon_images = list()
yrtpet_recon_images_np = list()
for iteration in iteration_range:
    if iteration + 1 == 7:
        yrtpet_recon_path = os.path.join(yrtpet_recon_dir, "recon_image.nii.gz")
    else:
        yrtpet_recon_path = os.path.join(
            yrtpet_recon_dir, f"recon_image_iteration{iteration+1}.nii.gz"
        )
    yrtpet_recon_paths.append(yrtpet_recon_path)
    yrtpet_recon_images.append(sitk.ReadImage(yrtpet_recon_path) * yrtpet_scale)
    yrtpet_recon_images_np.append(sitk.GetArrayViewFromImage(yrtpet_recon_images[-1]))


# %% Line plot

if plot_lineplot:

    # Apply decay and livetime corrections
    urt_recon_to_show = urt_recon_images[-1] * unit_scale
    molar_recon_to_show = molar_recon_images[-1] * unit_scale
    yrtpet_recon_to_show = yrtpet_recon_images[-1] * unit_scale

    xlim = (235, 395)
    ylim = (297, 467)

    fig_lineplot = ydisp.matshow_images_with_lineplot(
        [
            urt_recon_to_show,
            molar_recon_to_show,
            yrtpet_recon_to_show,
        ],
        zslice=460,
        x1=260,
        y1=414,
        x2=250,
        y2=363,
        xlim=xlim,
        ylim=ylim,
        labels=["URT", "MOLAR", "YRT-PET"],
        image_label_size=12,
        normalize=False,
        vmaxs=[800, 800, 800],
        fig_height=6,
        margin_bottom=0.08,
        margin_left=0.07,
        ylabel="Activity [kBq/mL]",
        linecolors=["#E69F00", "blue", "red"],
    )
    plt.savefig(os.path.join(figures_dir, "derenzo_lineplots.pdf"), dpi=600)
    plt.show()

# %% Convergence plot

if plot_convergence:

    pass
