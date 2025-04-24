#!/usr/bin/env python
import numpy as np
import SimpleITK as sitk
import os

roi_dir = "/data8/data/app311_tm_rhm/20250130_sm580/app311_brain/roi_qcs"
out_dir = "/data24/yn257/yrtpetnx_nssmic_sandbox/monkey"
yrtpet_recon_dir = "/data24/yn257/nx_monkey/recons/wbigvoxels_watt_wnorm_siddon_wdoi_wpsf_wtof_7i10s"
molar_recon_dir = "/data2/Recons/tm2225/20250130_sm580_baseline/injid_38594/listmode_001/"

# TODO: Read the region masks in roi_dir
# TODO: Resample the masks in the PET images
# TODO: Generate Time-Activity-Curves of the PET images

