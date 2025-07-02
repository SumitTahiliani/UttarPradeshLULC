import streamlit as st
st.set_page_config(layout="wide")

import requests
import os
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from shapely.geometry import box, mapping
import leafmap.foliumap as leafmap
import io
import tempfile
import json
from utils import buffer_bbox
from skimage.transform import resize

RASTER_FOLDER = "/home/azureuser/app/raster_data"

# === Constants ===
DW_CLASSES = {
    0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Vegetation', 4: 'Crops',
    5: 'Shrub & Scrub', 6: 'Built-up', 7: 'Bare Ground', 8: 'Snow & Ice'
}
VIS_CLASS_IDS = list(range(9))
VIS_PALETTE = [
    '#419bdf', '#397d49', '#88b053', '#7a87c6', '#e49635',
    '#dfc35a', '#c4281b', '#a59b8f', '#b39fe1'
]
cmap = ListedColormap(VIS_PALETTE)


# === Utility ===
def get_raster_path(year):
    local_path = os.path.join(RASTER_FOLDER, f"dw_up_{year}.tif")
    if not os.path.exists(local_path):
        st.warning(f"Raster file not found: {local_path}")
        return None
    return local_path

def downsample_and_mask(src, geojson_geom, scale_factor=0.2):
    try:
        out_image, _ = mask(src, geojson_geom, crop=True)
    except ValueError as e:
        st.warning("⚠️ AOI does not overlap raster extent.")
        st.write("Raster bounds:", src.bounds)
        st.write("AOI bounds:", geojson_geom)
        raise e
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

# === Caching ===
@st.cache_data
def search_location(query):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 5, "polygon_geojson": 1}
    headers = {"User-Agent": "streamlit-gis-app/1.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Nominatim API error: {e}")
        return []

@st.cache_data
def analyze_aoi(bbox):
    records = {}
    images_by_year = {}
    minx, miny, maxx, maxy = map(float, bbox)
    geojson_geom = [mapping(box(minx, miny, maxx, maxy))]

    for year in range(2016, 2026):
        path = get_raster_path(year)
        if not os.path.exists(path):
            continue

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

def plot_land_cover_trends(df):
    class_colors = dict(zip(DW_CLASSES.values(), VIS_PALETTE))
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '>']
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, class_name in enumerate(DW_CLASSES.values()):
        if class_name in df.columns:
            ax.plot(df.index, df[class_name], label=class_name, color=class_colors[class_name], marker=markers[i % len(markers)])

    ax.set(title="Land Cover Class Percentage Over Time", xlabel="Year", ylabel="Area (%)")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        st.download_button(
        label="📥 Download Trend Plot as PNG",
        data=buf,
        file_name="land_cover_trends.png",
        mime="image/png"
        )
def visualize_multiple_years(images_by_year):
    years = sorted(images_by_year)
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    norm = BoundaryNorm(VIS_CLASS_IDS + [VIS_CLASS_IDS[-1] + 1], cmap.N)

    for ax, year in zip(axs.ravel(), years):
        im = ax.imshow(images_by_year[year], cmap=cmap, norm=norm)
        ax.set(title=str(year), xticks=[], yticks=[])

    for ax in axs.ravel()[len(years):]:
        ax.axis('off')

    cbar = fig.colorbar(im, ax=axs, ticks=VIS_CLASS_IDS, fraction=0.03, pad=0.01)
    cbar.ax.set_yticklabels(list(DW_CLASSES.values()))
    cbar.set_label("Land Cover Class", rotation=270, labelpad=15)
    fig.tight_layout(); plt.subplots_adjust(right=0.88)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
    st.download_button(
            label="📥 Download Yearly Maps as PNG",
            data=buf,
            file_name="land_cover_maps.png",
            mime="image/png"
        )

def show_overlay_on_map(image_array, bbox):
    minx, miny, maxx, maxy = bbox
    height, width = image_array.shape
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    rgb = (ListedColormap(VIS_PALETTE)(image_array / 8.0)[:, :, :3] * 255).astype(np.uint8)
    rgb = np.transpose(rgb, (2, 0, 1))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_tif:
        with rasterio.open(temp_tif.name, "w", driver="GTiff", height=height, width=width, count=3, dtype=rgb.dtype, crs="EPSG:4326", transform=transform) as dst:
            for i in range(3): dst.write(rgb[i], i+1)

    m = leafmap.Map(center=[(miny+maxy)/2, (minx+maxx)/2], zoom=14)
    m.add_basemap("SATELLITE")
    m.add_raster(temp_tif.name, layer_name="Overlay", opacity=0.4)
    m.add_geojson(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]
            },
            "properties": {"name": "AOI"}
        }]
    }), layer_name="AOI")
    m.add_legend(title="Land Cover Classes", legend_dict={DW_CLASSES[i]: VIS_PALETTE[i] for i in range(9)})
    m.to_streamlit(height=550)

@st.cache_data
def get_array(year, geojson_geom):
    path = get_raster_path(year)
    with rasterio.open(path) as src:
        return downsample_and_mask(src, geojson_geom, scale_factor=0.2)

def display_change_detection(bbox):
    st.write("## 🔄 Change Detection (2016 → 2025)")
    geojson_geom = [mapping(box(*bbox))]

    try:
        arr_2016 = get_array(2016, geojson_geom)
        arr_2025 = get_array(2025, geojson_geom)
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
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.imshow(arr_2016 != arr_2025, cmap='gray')
        ax1.set_title("Changed Pixels", fontsize=14)
        ax1.axis('off')
        st.pyplot(fig1)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", bbox_inches='tight')
        buf1.seek(0)
        st.download_button(
            label="📥 Download Changed Pixels Map",
            data=buf1,
            file_name="changed_pixels.png",
            mime="image/png"
        )

    with plot_col2:
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.imshow(mask_change, cmap='Reds')
        ax2.set_title(f"{from_class} → {to_class}", fontsize=14)
        ax2.axis('off')
        st.pyplot(fig2)
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches='tight')
        buf2.seek(0)
        st.download_button(
            label=f"📥 Download {from_class}→{to_class} Change Map",
            data=buf2,
            file_name=f"{from_class.lower().replace(' ', '_')}_to_{to_class.lower().replace(' ', '_')}.png",
            mime="image/png"
        )

# === Streamlit App ===
st.title("Land Cover Analysis (Uttar Pradesh, Dynamic World)")
query = st.text_input("Search a location (in UP)", "")

if query and "selected_bbox" not in st.session_state:
    st.write("Suggestions:")
    for idx, loc in enumerate(search_location(query)):
        name = loc["display_name"]
        bbox = [float(loc["boundingbox"][2]), float(loc["boundingbox"][0]), float(loc["boundingbox"][3]), float(loc["boundingbox"][1])]
        with st.expander(name):
            if st.button(f"Analyze {name}", key=f"btn-{idx}"):
                buffered = buffer_bbox(bbox)
                st.session_state["selected_bbox"] = buffered
                st.session_state["location_name"] = name
                st.rerun()

if "selected_bbox" in st.session_state:
    buffered = st.session_state["selected_bbox"]
    st.write(f"Selected location: {st.session_state.get('location_name', '')}")
    st.write(f"Buffered bbox: {buffered}")

    # Persistent checkbox states
    for k, v in {"show_maps": True, "show_overlay": False, "show_change": False}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.session_state["show_maps"] = st.checkbox("Show Land Cover Maps Over 10 Years", value=st.session_state["show_maps"], key="show_maps_checkbox")
    st.session_state["show_overlay"] = st.checkbox("Show Satellite Overlay (2025)", value=st.session_state["show_overlay"], key="show_overlay_checkbox")
    st.session_state["show_change"] = st.checkbox("Enable Change Detection", value=st.session_state["show_change"], key="show_change_checkbox")

    with st.spinner("Running analysis..."):
        df, images = analyze_aoi(buffered)
        if not df.empty:
            st.write("### Land Cover Class Trends Over Time")
            # st.dataframe(df, use_container_width=True, height=300)
            plot_land_cover_trends(df)
            if st.session_state["show_maps"]:
                st.write("### Land Cover Maps Over 10 Years")
                visualize_multiple_years(images)

            if st.session_state["show_overlay"]:
                tif_path = get_raster_path(2025)
                if tif_path:
                    with rasterio.open(tif_path) as src:
                        out_image, _ = mask(src, [mapping(box(*buffered))], crop=True)
                        st.write("### Landcover Map over Satellite Imagery (2025)")
                        show_overlay_on_map(out_image[0], buffered)

            if st.session_state["show_change"]:
                with st.expander("Change Detection", expanded=True):
                    display_change_detection(buffered)
        else:
            st.warning("No data found for this region.")
