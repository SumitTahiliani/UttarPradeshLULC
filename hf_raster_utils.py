"""
Raster I/O and masking utilities for LULC project (HuggingFace version, multi-state).
Handles downloading rasters from HuggingFace and caching them locally.
"""
import os
import numpy as np
import rasterio
from rasterio.mask import mask
from skimage.transform import resize
from typing import Optional, List
import requests
import streamlit as st

@st.cache_data
def get_raster_path(state_code: str, year: int) -> Optional[str]:
    """Download and cache raster for a given state and year from HuggingFace, return local path."""
    folder = f"dw_{state_code}_rasters"
    filename = f"dw_{state_code}_{year}.tif"
    base_url = f"https://huggingface.co/datasets/sumittahiliani/UttarPradeshYearlyDynamicWorld/resolve/main/{folder}/"
    download_url = base_url + filename
    cache_dir = os.path.join(os.getcwd(), folder)
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_path):
        try:
            st.info(f"Downloading raster for {state_code.upper()}, {year}...")
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            downloaded_size = 0
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress_bar.progress(min(downloaded_size / total_size, 1.0))
            progress_bar.empty()
        except Exception as e:
            st.warning(f"Failed to download raster for {state_code.upper()}, {year}: {e}")
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