#!/usr/bin/env python
import numpy as np
import os
import glob

def get_molar_image_path(molar_dir: str, frame: int, iteration: int = 7):
    molar_recon_image_glob_path = os.path.join(
        molar_dir, f"Frame{str(frame+1).zfill(3)}", f"*-{iteration}.img"
    )
    molar_recon_image_glob = glob.glob(molar_recon_image_glob_path)
    if len(molar_recon_image_glob) > 1:
        raise RuntimeError(
            f"There is more than one MOLAR recon file in the folder for frame {frame}"
        )
    if len(molar_recon_image_glob) < 1:
        raise RuntimeError(
            f"There is no MOLAR recon file in the folder for frame {frame}"
        )
    molar_recon_image_path = molar_recon_image_glob[0]
    return molar_recon_image_path

def get_molar_decay_and_livetime_factors(molar_recon_dir:str, num_frames:int):
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
    return (decay_factors, livetime_factors)
