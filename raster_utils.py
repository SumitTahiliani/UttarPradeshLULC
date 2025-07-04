"""
Raster I/O and masking utilities for Uttar Pradesh LULC project and other supported states.
"""
import os
import numpy as np
import rasterio
from rasterio.mask import mask
from skimage.transform import resize
from typing import Optional, List
import streamlit as st

# Mapping from state codes to actual filename patterns used in folders
STATE_FILENAME_PATTERNS = {
    "up": "dw_up_{year}.tif",
    "mh": "dw_mh_{year}.tif",
    "wb": "dw_westbengal_{year}.tif"
}

def get_raster_path(state_code: str, year: int) -> Optional[str]:
    """Return the local raster file path for a given state and year, or None if not found."""
    folder = f"dw_{state_code}_rasters"
    filename_pattern = STATE_FILENAME_PATTERNS.get(state_code)
    if not filename_pattern:
        return None
    filename = filename_pattern.format(year=year)
    local_path = os.path.join(folder, filename)
    if not os.path.exists(local_path):
        return None
    return local_path

def downsample_and_mask(src, geojson_geom, scale_factor: float = 0.2) -> np.ndarray:
    """Mask and optionally downsample a rasterio dataset using a geojson geometry."""
    out_image, _ = mask(src, geojson_geom, crop=True)
    data = out_image[0]
    if scale_factor != 1.0:
        data = resize(
            data,
            (int(data.shape[0] * scale_factor), int(data.shape[1] * scale_factor)),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.uint8)
    return data 