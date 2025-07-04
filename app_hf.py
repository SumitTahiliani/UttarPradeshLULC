"""
Streamlit UI for LULC project (multi-state, HuggingFace rasters).
Handles user interaction and high-level orchestration only.
"""

import streamlit as st
from utils import buffer_bbox, detect_state_from_bbox
from hf_raster_utils import get_raster_path, downsample_and_mask
from analysis import analyze_aoi, get_array, display_change_detection
from plotting import plot_land_cover_trends, visualize_multiple_years, show_overlay_on_map
from geocoding import search_location

import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import pandas as pd

DW_CLASSES = {
    0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Vegetation', 4: 'Crops',
    5: 'Shrub & Scrub', 6: 'Built-up', 7: 'Bare Ground', 8: 'Snow & Ice'
}

st.set_page_config(layout="wide")

st.title("Land Cover Analysis (India, Dynamic World, HuggingFace)")

# New Analysis button
if "selected_bbox" in st.session_state:
    if st.button("🔄 New Analysis", type="primary"):
        # Clear all session state variables
        for key in ["selected_bbox", "location_name", "state_code", "state_name", "show_maps", "show_overlay", "show_change"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

query = st.text_input("Search a location (in India)", "")

if query and "selected_bbox" not in st.session_state:
    st.write("Suggestions:")
    for idx, loc in enumerate(search_location(query)):
        name = loc["display_name"]
        bbox = [float(loc["boundingbox"][2]), float(loc["boundingbox"][0]), float(loc["boundingbox"][3]), float(loc["boundingbox"][1])]
        state_name, state_code = detect_state_from_bbox(bbox)
        with st.expander(name):
            st.write(f"Detected state: {state_name if state_name else 'Unknown'}")
            if not state_code:
                st.warning("Location not in a supported state.")
            else:
                if st.button(f"Analyze {name}", key=f"btn-{idx}"):
                    buffered = buffer_bbox(bbox)
                    st.session_state["selected_bbox"] = buffered
                    st.session_state["location_name"] = name
                    st.session_state["state_code"] = state_code
                    st.session_state["state_name"] = state_name
                    st.rerun()

if "selected_bbox" in st.session_state and "state_code" in st.session_state:
    buffered = st.session_state["selected_bbox"]
    state_code = st.session_state["state_code"]
    state_name = st.session_state.get("state_name", state_code)
    st.write(f"Selected location: {st.session_state.get('location_name', '')}")
    st.write(f"Detected state: {state_name}")
    st.write(f"Buffered bbox: {buffered}")
    for k, v in {"show_maps": True, "show_overlay": False, "show_change": False}.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state["show_maps"] = st.checkbox("Show Land Cover Maps Over 10 Years", value=st.session_state["show_maps"], key="show_maps_checkbox")
    st.session_state["show_overlay"] = st.checkbox("Show Satellite Overlay (2025)", value=st.session_state["show_overlay"], key="show_overlay_checkbox")
    st.session_state["show_change"] = st.checkbox("Enable Change Detection", value=st.session_state["show_change"], key="show_change_checkbox")
    with st.spinner("Running analysis..."):
        df, images = analyze_aoi(state_code, buffered)
        if not df.empty:
            st.write("### Land Cover Class Trends Over Time")
            plot_land_cover_trends(df)
            if st.session_state["show_maps"]:
                st.write("### Land Cover Maps Over 10 Years")
                visualize_multiple_years(images)
            if st.session_state["show_overlay"]:
                tif_path = get_raster_path(state_code, 2025)
                if tif_path:
                    try:
                        with rasterio.open(tif_path) as src:
                            out_image, _ = mask(src, [mapping(box(*buffered))], crop=True)
                            st.write("### Landcover Map over Satellite Imagery (2025)")
                            show_overlay_on_map(out_image[0], buffered)
                    except ValueError as e:
                        if "Input shapes do not overlap raster" in str(e) and state_code == "mh":
                            # Use 2024 data as fallback for Maharashtra 2025
                            tif_path_2024 = get_raster_path(state_code, 2024)
                            if tif_path_2024:
                                with rasterio.open(tif_path_2024) as src:
                                    out_image, _ = mask(src, [mapping(box(*buffered))], crop=True)
                                    st.write("### Landcover Map over Satellite Imagery (2024 - 2025 data unavailable)")
                                    show_overlay_on_map(out_image[0], buffered)
                        else:
                            st.warning("Unable to load overlay data.")
            if st.session_state["show_change"]:
                with st.expander("Change Detection", expanded=True):
                    display_change_detection(state_code, buffered)
        else:
            st.warning("No data found for this region.")
