"""
Analysis functions for AOI and change detection for LULC project (multi-state).
"""
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box, mapping
import streamlit as st
from raster_utils import get_raster_path, downsample_and_mask
from typing import Dict, Tuple, List, Optional
from plotting import show_overlay_on_map
import os

DW_CLASSES = {
    0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Vegetation', 4: 'Crops',
    5: 'Shrub & Scrub', 6: 'Built-up', 7: 'Bare Ground', 8: 'Snow & Ice'
}

@st.cache_data
def analyze_aoi(state_code: str, bbox: List[float]) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """Analyze AOI for all years and return a DataFrame and images by year for the given state."""
    records = {}
    images_by_year = {}
    minx, miny, maxx, maxy = map(float, bbox)
    geojson_geom = [mapping(box(minx, miny, maxx, maxy))]
    
    for year in range(2016, 2026):
        path = get_raster_path(state_code, year)
        if not path or not os.path.exists(path):
            continue
        
        # Special handling for Maharashtra 2025 (use 2024 as fallback)
        if state_code == "mh" and year == 2025:
            try:
                with rasterio.open(path) as src:
                    data = downsample_and_mask(src, geojson_geom, scale_factor=0.2)
            except ValueError as e:
                if "Input shapes do not overlap raster" in str(e):
                    st.warning("⚠️ Maharashtra 2025 raster has limited extent. Using 2024 data instead.")
                    # Try to use 2024 data for 2025
                    path_2024 = get_raster_path(state_code, 2024)
                    if path_2024 and os.path.exists(path_2024):
                        with rasterio.open(path_2024) as src:
                            data = downsample_and_mask(src, geojson_geom, scale_factor=0.2)
                    else:
                        continue
                else:
                    raise e
        else:
            with rasterio.open(path) as src:
                data = downsample_and_mask(src, geojson_geom, scale_factor=0.2)
        
        if src.nodata is not None:
            data = data[data != src.nodata]
        unique, counts = np.unique(data, return_counts=True)
        total = counts.sum()
        records[year] = {
            DW_CLASSES.get(int(cls), str(cls)): round(cnt/total*100, 2)
            for cls, cnt in zip(unique, counts) if cls in DW_CLASSES
        }
        records[year]["Year"] = year
        images_by_year[year] = data
    df = pd.DataFrame.from_dict(records, orient='index').sort_values("Year").set_index("Year")
    return df, images_by_year

@st.cache_data
def get_array(state_code: str, year: int, geojson_geom: List[dict]) -> np.ndarray:
    """Get masked and downsampled array for a given state, year, and geometry."""
    path = get_raster_path(state_code, year)
    
    # Special handling for Maharashtra 2025
    if state_code == "mh" and year == 2025:
        try:
            with rasterio.open(path) as src:
                return downsample_and_mask(src, geojson_geom, scale_factor=0.2)
        except ValueError as e:
            if "Input shapes do not overlap raster" in str(e):
                # Use 2024 data as fallback
                path_2024 = get_raster_path(state_code, 2024)
                if path_2024 and os.path.exists(path_2024):
                    with rasterio.open(path_2024) as src:
                        return downsample_and_mask(src, geojson_geom, scale_factor=0.2)
            raise e
    
    with rasterio.open(path) as src:
        return downsample_and_mask(src, geojson_geom, scale_factor=0.2)

def display_change_detection(state_code: str, bbox: List[float]) -> None:
    """Display change detection between 2016 and 2025 for the AOI and state."""
    import matplotlib.pyplot as plt
    import io

    st.write("## 🔄 Change Detection (2016 → 2025)")
    geojson_geom = [mapping(box(*bbox))]

    try:
        arr_2016 = get_array(state_code, 2016, geojson_geom)
        arr_2025 = get_array(state_code, 2025, geojson_geom)
        if arr_2016.shape != arr_2025.shape:
            st.warning("Masked arrays differ in shape.")
            return
    except Exception as e:
        st.warning(f"Error: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        from_class = st.selectbox("From Class", list(DW_CLASSES.values()), index=1, key="from_class")
    with col2:
        to_class = st.selectbox("To Class", list(DW_CLASSES.values()), index=6, key="to_class")

    from_id = list(DW_CLASSES.keys())[list(DW_CLASSES.values()).index(from_class)]
    to_id = list(DW_CLASSES.keys())[list(DW_CLASSES.values()).index(to_class)]
    mask_change = (arr_2016 == from_id) & (arr_2025 == to_id)

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        fig1, ax1 = plt.subplots(figsize=(3, 3), dpi=100)
        ax1.imshow(arr_2016 != arr_2025, cmap='gray', aspect='auto')
        ax1.set_title("Changed Pixels", fontsize=12)
        ax1.axis('off')
        fig1.tight_layout(pad=0)
        st.pyplot(fig1)

        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", dpi=100)
        buf1.seek(0)
        st.download_button(
            label="📥 Download Changed Pixels Map",
            data=buf1,
            file_name="changed_pixels.png",
            mime="image/png"
        )

    with plot_col2:
        fig2, ax2 = plt.subplots(figsize=(3, 3), dpi=100)
        ax2.imshow(mask_change, cmap='Reds', aspect='auto')
        ax2.set_title(f"{from_class} → {to_class}", fontsize=12)
        ax2.axis('off')
        fig2.tight_layout(pad=0)
        st.pyplot(fig2)

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=100)
        buf2.seek(0)
        st.download_button(
            label=f"📥 Download {from_class}→{to_class} Change Map",
            data=buf2,
            file_name=f"{from_class.lower().replace(' ', '_')}_to_{to_class.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
