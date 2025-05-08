#!/usr/bin/env python
import numpy as np
import SimpleITK as sitk
import yn_tools.kinetic as ykin
import os
import matplotlib.pyplot as plt
import python_tools.plot as ptplt
from molar_helper import *
import skimage.measure as skm

# %% Constants

yrtpet_recon_dir = "/data24/yn257/nx_human_20231002_RR584/recons/watt_wnorm_siddon_wdoi_wpsf_wtof_7i10s"
molar_recon_dir = (
    "/data2/Recons/kf283/2025_mic/yrt/20231002_RR584/injid_34473/listmode_001"
)
figures_dir = "/home1/yn257/work/data/yrtpetnx_nssmic/figures"
unit_scale = 1.0 / 1000.0  # Bq to kBq
molar_global = 3.188e-9
frames_to_show = [3, 9, 15, 25]
frame_of_lineplot = 15

# %% Defining the frame timings

frame_timing = [(6, 30), (3, 60), (2, 120), (22, 300)]
frame_durations = ykin.frame_timing_to_frame_durations(frame_timing)
frame_ranges = ykin.frame_durations_to_frame_ranges(frame_durations)
frame_ranges = [(int(f[0]), int(f[1])) for f in frame_ranges]  # Make sure it's integers
num_frames = len(frame_ranges)

frame_ranges_decomposed = [
    (ykin.decompose_seconds(f[0]), ykin.decompose_seconds(f[1])) for f in frame_ranges
]

# %% Compute scale factor

decay_factors, livetime_factors = get_molar_decay_and_livetime_factors(
    molar_recon_dir, num_frames
)

# %% Read images from YRT-PET

yrtpet_recon_images = list()
molar_recon_images = list()

for frame_i in frames_to_show:

    yrtpet_recon_path = os.path.join(
        yrtpet_recon_dir, f"frame_{str(frame_i).zfill(2)}", "recon_image.nii.gz"
    )
    molar_recon_path = get_molar_image_path(molar_recon_dir, frame_i)

    molar_recon_image = sitk.ReadImage(molar_recon_path)
    yrtpet_recon_image = (
        sitk.ReadImage(yrtpet_recon_path)
        / (molar_global * frame_durations[frame_i])
        / decay_factors[frame_i]
        / livetime_factors[frame_i]
    )
    yrtpet_recon_image = sitk.Cast(yrtpet_recon_image, sitk.sitkFloat32)
    molar_recon_image.SetOrigin(yrtpet_recon_image.GetOrigin())

    molar_recon_images.append(molar_recon_image)
    yrtpet_recon_images.append(yrtpet_recon_image)

voxel_size = [1.0, 0.8, 0.8]


# %% Generate figure

vranges = [[0, 50], [0, 50]]
label_list = ["MOLAR", "YRT-PET"]
colormaps = ["gray", "gray"]
slice_x = 336

linecolors = ["blue", "red"]
linewidth = 1
lineplot_coords = 351, 179, 233, 210  # x1, y1, x2, y2
line_profiles = list()
lineplot_ax_frac = 1.2

# [x0, x1, y0, y1]
cbox = [210, 420, 165, 335]
colorbar_width_ratio = 0.4

h_ims = list()

fig_width = 6
fig_slack_factors = [75, 0]  # x, y
fig_height = (
    fig_width
    / (
        len(frames_to_show) * (cbox[1] - cbox[0] + 1) * voxel_size[1]
        + fig_slack_factors[0]
    )
    * (
        (len(label_list) + lineplot_ax_frac) * (cbox[3] - cbox[2] + 1) * voxel_size[0]
        + fig_slack_factors[1]
    )
)

fig = plt.figure()
fig.set_size_inches([fig_width, fig_height])

nrows = len(label_list) + 1  # +1 for lineplot
ncols = len(frames_to_show)

grid = fig.add_gridspec(
    nrows,
    ncols,
    height_ratios=[
        1,
        1,
        lineplot_ax_frac,
    ],
    width_ratios=[1] * len(frames_to_show),
)

fig.subplots_adjust(
    left=0.1, right=0.88, top=0.95, bottom=0.1, hspace=0.01, wspace=0.01
)

axes_img = list()
for i in range(len(label_list)):
    curr_axes_img = list()
    for j in range(len(frames_to_show)):
        curr_axes_img.append(fig.add_subplot(grid[i, j]))
    axes_img.append(curr_axes_img)


recon_images_groups = [molar_recon_images, yrtpet_recon_images]

for lbl, ax_l, recon_images, vrange, colormap, linecolor in zip(
    label_list, axes_img, recon_images_groups, vranges, colormaps, linecolors
):
    for i, fr, ax in zip(range(len(frames_to_show)), frames_to_show, ax_l):

        (hour_start, minute_start, second_start), (hour_end, minute_end, second_end) = (
            frame_ranges_decomposed[fr]
        )

        img_sitk = recon_images[i]
        img_np = sitk.GetArrayFromImage(img_sitk) * unit_scale
        img_slice_np = img_np[:, :, slice_x]
        h_ims.append(
            ax.matshow(img_slice_np, vmin=vrange[0], vmax=vrange[1], cmap=colormap)
        )
        ax.set_axis_off()
        ax.set_xlim(cbox[:2][::-1])
        ax.set_ylim(cbox[2:][::-1])
        ax.set_aspect(img_sitk.GetSpacing()[2] / img_sitk.GetSpacing()[1])

        if hour_start != 0 or hour_end != 0:
            text_label = (
                f"{hour_start:01d}:{minute_start:02d}:{second_start:02d} to "
                + f"{hour_end:01d}:{minute_end:02d}:{second_end:02d}"
            )
        else:
            text_label = (
                f"{minute_start:02d}:{second_start:02d} to "
                + f"{minute_end:02d}:{second_end:02d}"
            )
        col_label = "({})".format(chr(ord("a") + frames_to_show.index(fr)))
        if lbl == label_list[0]:
            print(col_label + " from " + text_label)
            ax.text(
                np.mean(ax.get_xlim()),
                np.min(ax.get_ylim()),
                text_label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        if fr == frame_of_lineplot:
            x1, y1, x2, y2 = lineplot_coords

            line_profile = skm.profile_line(img_slice_np, (y1, x1), (y2, x2))
            line_profiles.append(line_profile)

            # Draw two lines, one white behind and one colored in front
            for [curr_linecolor, curr_linewidth] in zip(
                ["white", linecolor], [linewidth * 1.5, linewidth]
            ):
                ax.annotate(
                    "",  # No text
                    xy=(x2, y2),  # End of the arrow
                    xytext=(x1, y1),  # Start of the arrow
                    arrowprops=dict(
                        arrowstyle="->",  # Simple arrow
                        lw=curr_linewidth,  # Line width
                        color=curr_linecolor,  # Arrow color
                    ),
                )
    ax_l[0].text(
        np.max(ax_l[0].get_xlim()),
        np.mean(ax_l[0].get_ylim()),
        lbl,
        ha="right",
        va="center",
        fontsize=11,
        rotation=90,
    )

ax_ref_list = [ax_l[-1] for ax_l in axes_img]

# Add colorbar
cb_pos = ptplt.get_ax_pos_rel(
    ax_ref_list=ax_ref_list,
    pos="right",
    offset_frac=0.25,
    width_frac=0.1,
    height_frac=0.8,
)
cb_ax = fig.add_axes(
    [cb_pos[0], cb_pos[1], cb_pos[2] - cb_pos[0], cb_pos[3] - cb_pos[1]]
)
fig.colorbar(h_ims[0], cax=cb_ax)
cb_ax.set_title("[kBq/mL]", fontsize=9)

# Add line plots
line_profile_distance_y = (lineplot_coords[2] - lineplot_coords[0]) * voxel_size[1]
line_profile_distance_z = (lineplot_coords[3] - lineplot_coords[1]) * voxel_size[0]
line_profile_distance = np.sqrt(line_profile_distance_y**2 + line_profile_distance_z**2)

num = max([line_profiles[i].shape[0] for i in range(len(line_profiles))])
line_profile_distance_linspace = np.linspace(0, line_profile_distance, num)
ax_lineplot = fig.add_subplot(grid[-1, :])

for line_profile, label, linecolor in zip(line_profiles, label_list, linecolors):
    ax_lineplot.plot(
        line_profile_distance_linspace, line_profile, label=label, color=linecolor
    )
ax_lineplot.set_ylabel("Activity [kBq/mL]")
ax_lineplot.set_xlabel("Distance [mm]")
ax_lineplot.set_xlim((0, np.max(line_profile_distance_linspace)))
ax_lineplot.set_ylim((0, None))
ax_lineplot.legend(fontsize=8)


plt.savefig(os.path.join(figures_dir, "human_slices.pdf"), dpi=600)
plt.show()
