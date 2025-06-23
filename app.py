# import streamlit as st
# import requests
# import os
# import rasterio
# from rasterio.mask import mask
# from rasterio.transform import from_bounds
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib.colors import ListedColormap, BoundaryNorm
# from shapely.geometry import box, mapping
# import leafmap.foliumap as leafmap
# import tempfile
# import json
# from utils import buffer_bbox

# st.set_page_config(layout="wide")

# # === Constants ===
# DW_CLASSES = {
#     0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Vegetation', 4: 'Crops',
#     5: 'Shrub & Scrub', 6: 'Built-up', 7: 'Bare Ground', 8: 'Snow & Ice'
# }
# RASTER_FOLDER = "dw_up_rasters"

# # === Visualisation constants and functions ===
# VIS_CLASS_IDS = list(range(9))
# VIS_PALETTE = [
#     '#419bdf', '#397d49', '#88b053', '#7a87c6', '#e49635',
#     '#dfc35a', '#c4281b', '#a59b8f', '#b39fe1'
# ]
# cmap = ListedColormap(VIS_PALETTE)

# # === Nominatim API for location search ===
# def search_location(query):
#     url = "https://nominatim.openstreetmap.org/search"
#     params = {
#         "q": query,
#         "format": "json",
#         "limit": 5,
#         "polygon_geojson": 1
#     }
#     headers = {
#         "User-Agent": "streamlit-gis-app/1.0 (sumittahiliani24@gmail.com)"
#     }
#     try:
#         response = requests.get(url, params=params, headers=headers, timeout=10)
#         response.raise_for_status()  # raises HTTPError for 4xx/5xx
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         st.error(f"Nominatim API error: {e}")
#         return []
#     except ValueError as e:
#         st.error(f"Failed to decode response: {e}")
#         return []

# def visualize_multiple_years(images_by_year, CLASS_IDS, CLASS_NAMES, CLASS_COLORS):
#     years = sorted(images_by_year.keys())
#     nrows, ncols = 2,5
#     fig, axs = plt.subplots(nrows, ncols, figsize=(20, 8))  # Was (8, 16)

#     cmap = ListedColormap(CLASS_COLORS)
#     norm = BoundaryNorm(CLASS_IDS + [CLASS_IDS[-1] + 1], cmap.N)

#     for ax, year in zip(axs.ravel(), years):
#         image = images_by_year[year]
#         im = ax.imshow(image, cmap=cmap, norm=norm)
#         ax.set_title(str(year), fontsize=9)
#         ax.axis('off')

#     for ax in axs.ravel()[len(years):]:
#         ax.axis('off')

#     cbar = fig.colorbar(im, ax=axs, ticks=CLASS_IDS, fraction=0.03, pad=0.01)
#     cbar.ax.set_yticklabels(CLASS_NAMES)
#     cbar.set_label("Land Cover Class", rotation=270, labelpad=15)

#     plt.tight_layout()
#     plt.subplots_adjust(right=0.88)

#     st.pyplot(fig)

# def visualize_tif(image, year, cmap, class_id=VIS_CLASS_IDS,class_name=DW_CLASSES):

#     fig, ax = plt.subplots(figsize=(8, 8))
#     im = ax.imshow(image, cmap=cmap, vmin=0, vmax=8)
#     ax.set_title(f"Dynamic World Land Cover - {year}")
#     cbar = fig.colorbar(im, ticks=class_id, fraction=0.046, pad=0.04)
#     cbar.ax.set_yticklabels(class_name)
#     ax.axis('off')
#     fig.tight_layout()
    
#     st.pyplot(fig)
# def show_overlay_on_map(image_array, bbox, title="Overlay"):
#     import tempfile
#     import rasterio
#     from rasterio.transform import from_bounds
#     from rasterio.enums import ColorInterp
#     from matplotlib.colors import ListedColormap
#     import numpy as np
#     import leafmap.foliumap as leafmap

#     minx, miny, maxx, maxy = bbox
#     height, width = image_array.shape
#     transform = from_bounds(minx, miny, maxx, maxy, width, height)

#     # Apply color map manually
#     cmap = ListedColormap(VIS_PALETTE)
#     rgb_image = cmap(image_array / 8.0)[:, :, :3]  # Drop alpha
#     rgb_image = (rgb_image * 255).astype(np.uint8)

#     # Rearrange for rasterio (bands, height, width)
#     rgb_image = np.transpose(rgb_image, (2, 0, 1))

#     temp_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")

#     with rasterio.open(
#         temp_tif.name,
#         "w",
#         driver="GTiff",
#         height=height,
#         width=width,
#         count=3,
#         dtype=rgb_image.dtype,
#         crs="EPSG:4326",
#         transform=transform
#     ) as dst:
#         dst.write(rgb_image[0], 1)
#         dst.write(rgb_image[1], 2)
#         dst.write(rgb_image[2], 3)
#         dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]

#     # Display map
#     m = leafmap.Map(center=[(miny + maxy) / 2, (minx + maxx) / 2], zoom=14)
#     m.add_basemap("SATELLITE")
#     m.add_raster(temp_tif.name, layer_name=title, opacity=0.7)

#     # Add AOI
#     geojson_rectangle = {
#         "type": "FeatureCollection",
#         "features": [ {
#             "type": "Feature",
#             "geometry": {
#                 "type": "Polygon",
#                 "coordinates": [[
#                     [minx, miny], [maxx, miny],
#                     [maxx, maxy], [minx, maxy],
#                     [minx, miny]
#                 ]]
#             },
#             "properties": {"name": "AOI"}
#         }]
#     }
#     m.add_geojson(json.dumps(geojson_rectangle), layer_name="AOI")

#     # Add legend manually
#     legend_dict = {DW_CLASSES[i]: VIS_PALETTE[i] for i in range(len(VIS_PALETTE))}
#     m.add_legend(title="Land Cover Classes", legend_dict=legend_dict)

#     m.to_streamlit(height=550)

# # def show_overlay_on_map(image_array, bbox, title="Overlay"):
# #     import matplotlib.pyplot as plt
# #     from matplotlib.colors import ListedColormap
# #     import tempfile

# #     # Save to temporary GeoTIFF
# #     temp_tif = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
# #     minx, miny, maxx, maxy = bbox
# #     height, width = image_array.shape
# #     transform = from_bounds(minx, miny, maxx, maxy, width, height)

# #     with rasterio.open(
# #         temp_tif.name,
# #         "w",
# #         driver="GTiff",
# #         height=height,
# #         width=width,
# #         count=1,
# #         dtype=image_array.dtype,
# #         crs="EPSG:4326",
# #         transform=transform
# #     ) as dst:
# #         dst.write(image_array, 1)

# #     # Create map
# #     m = leafmap.Map(center=[(miny + maxy) / 2, (minx + maxx) / 2], zoom=14)
# #     m.add_basemap("SATELLITE")

# #     # Create colormap dict from your palette
# #     colormap_dict = {i: VIS_PALETTE[i] for i in range(len(VIS_PALETTE))}

# #     m.add_raster(temp_tif.name, layer_name=title, colormap=colormap_dict, nodata=255, opacity=0.6)

# #     # Add AOI
# #     geojson_rectangle = {
# #         "type": "FeatureCollection",
# #         "features": [{
# #             "type": "Feature",
# #             "geometry": {
# #                 "type": "Polygon",
# #                 "coordinates": [[
# #                     [minx, miny], [maxx, miny],
# #                     [maxx, maxy], [minx, maxy],
# #                     [minx, miny]
# #                 ]]
# #             },
# #             "properties": {"name": "AOI"}
# #         }]
# #     }
# #     m.add_geojson(json.dumps(geojson_rectangle), layer_name="AOI")

# #     # Add custom legend
# #     legend_dict = {DW_CLASSES[i]: VIS_PALETTE[i] for i in range(len(VIS_PALETTE))}
# #     m.add_legend(title="Land Cover Classes", legend_dict=legend_dict)

# #     m.to_streamlit(height=500)


# def analyze_aoi(bbox):
#     results = {}
#     images_by_year = {}
#     records = []

#     minx, miny, maxx, maxy = map(float, bbox)
#     bbox_geom = box(minx, miny, maxx, maxy)
#     geojson_geom = [mapping(bbox_geom)]

#     for year in range(2016, 2026):
#         path = os.path.join(RASTER_FOLDER, f"dw_up_{year}.tif")
#         if not os.path.exists(path):
#             continue

#         with rasterio.open(path) as src:
#             out_image, out_transform = mask(src, geojson_geom, crop=True)
#             data = out_image[0]

#             # Ignore nodata values
#             if src.nodata is not None:
#                 data = data[data != src.nodata]

#             # Flatten and count
#             unique, counts = np.unique(data, return_counts=True)
#             total = counts.sum()

#             class_percentages = {
#                 DW_CLASSES.get(int(cls), str(cls)): round(count / total * 100, 2)
#                 for cls, count in zip(unique, counts)
#                 if cls in DW_CLASSES
#             }

#             class_percentages["Year"] = year
#             records.append(class_percentages)
#             images_by_year[year] = out_image[0]
#             results[year] = class_percentages

#     df = pd.DataFrame(records).sort_values("Year").set_index("Year")
#     df.to_csv("class_percentage_trends.csv")

#     # Optional display/plot
#     st.dataframe(df)
#     return df, images_by_year

# # === Plotting ===
# def plot_land_cover_trends(df):
#     CLASS_NAMES = [
#         'Water', 'Trees', 'Grass', 'Flooded Vegetation', 'Crops',
#         'Shrub & Scrub', 'Built-up', 'Bare Ground', 'Snow & Ice'
#     ]
    
#     CLASS_COLORS = {
#         'Water': '#419bdf',
#         'Trees': '#397d49',
#         'Grass': '#88b053',
#         'Flooded Vegetation': '#7a87c6',
#         'Crops': '#e49635',
#         'Shrub & Scrub': '#dfc35a',
#         'Built-up': '#c4281b',
#         'Bare Ground': '#a59b8f',
#         'Snow & Ice': '#b39fe1'
#     }

#     MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '>']

#     fig, ax = plt.subplots(figsize=(14, 7))

#     for i, class_name in enumerate(CLASS_NAMES):
#         if class_name in df.columns:
#             ax.plot(
#                 df.index, df[class_name],
#                 label=class_name,
#                 color=CLASS_COLORS[class_name],
#                 marker=MARKERS[i % len(MARKERS)]
#             )

#     ax.set_title("Land Cover Class Percentage Over Time (Dynamic World)")
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Area (%)")
#     ax.grid(True)
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     fig.tight_layout()

#     st.pyplot(fig)

# def display_change_detection(buffered_bbox, raster_dir, class_names, scale_factor=1.0):
#     from shapely.geometry import box, mapping

#     st.write("## 🔄 Change Detection (2016 → 2025)")

#     bbox_geom = box(*buffered_bbox)
#     geojson_geom = [mapping(bbox_geom)]

#     def mask_and_downsample(tif_path):
#         with rasterio.open(tif_path) as src:
#             out_image, _ = mask(src, geojson_geom, crop=True)
#             return out_image[0]

#     try:
#         arr_2016 = mask_and_downsample(os.path.join(raster_dir, 'dw_up_2016.tif'))
#         arr_2025 = mask_and_downsample(os.path.join(raster_dir, 'dw_up_2025.tif'))
#     except Exception as e:
#         st.warning(f"Error during AOI clipping: {e}")
#         return

#     # Ensure same shape
#     if arr_2016.shape != arr_2025.shape:
#         st.warning("Masked arrays differ in shape.")
#         return

#     change_mask = arr_2016 != arr_2025
#     fig1, ax1 = plt.subplots(figsize=(6, 6))
#     ax1.imshow(change_mask, cmap='gray')
#     ax1.set_title("All Changed Pixels (2016 → 2025)")
#     ax1.axis('off')
#     st.pyplot(fig1)

#     # Specific transition
#     col1, col2 = st.columns(2)
#     with col1:
#         from_class = st.selectbox("From Class", class_names, index=class_names.index("Trees"))
#     with col2:
#         to_class = st.selectbox("To Class", class_names, index=class_names.index("Built-up"))

#     from_id = class_names.index(from_class)
#     to_id = class_names.index(to_class)

#     specific_mask = (arr_2016 == from_id) & (arr_2025 == to_id)
#     fig2, ax2 = plt.subplots(figsize=(6, 6))
#     ax2.imshow(specific_mask, cmap='Reds')
#     ax2.set_title(f"{from_class} → {to_class} (2016 → 2025)")
#     ax2.axis('off')
#     st.pyplot(fig2)



# # === Streamlit App ===
# st.title("Land Cover Analysis (Uttar Pradesh, Dynamic World)")
# query = st.text_input("Search a location (in UP)", "")

# if "selected_bbox" in st.session_state:
#     buffered_bbox = st.session_state["selected_bbox"]
#     name = st.session_state.get("location_name", "Selected Area")

#     st.write(f"Selected location: {name}")
#     st.write(f"Buffered bbox: {buffered_bbox}")
#     with st.spinner("Running analysis..."):
#         results, results_images = analyze_aoi(buffered_bbox)
#         if not results.empty:
#             st.write("### Land Cover Class Trends Over Time")
#             plot_land_cover_trends(results)
#             st.write("### Land Cover Maps Over 10 Years")
#             visualize_multiple_years(results_images, VIS_CLASS_IDS, list(DW_CLASSES.values()), VIS_PALETTE)
            
#             # Show satellite overlay
#             tif_path = os.path.join(RASTER_FOLDER, "dw_up_2025.tif")
#             if os.path.exists(tif_path):
#                 with rasterio.open(tif_path) as src:
#                     geojson_geom = [mapping(box(*buffered_bbox))]
#                     out_image, _ = mask(src, geojson_geom, crop=True)
#                     show_overlay_on_map(out_image[0], buffered_bbox)

#             # Change detection
#             if os.path.exists(os.path.join(RASTER_FOLDER, 'dw_up_2016.tif')) and \
#                os.path.exists(os.path.join(RASTER_FOLDER, 'dw_up_2025.tif')):
#                 display_change_detection(buffered_bbox, RASTER_FOLDER, list(DW_CLASSES.values()))

#             st.success("Done!")
#         else:
#             st.warning("No data found for this region.")
# if query:
#     st.write("Suggestions:")
#     locations = search_location(query)
#     for idx, loc in enumerate(locations):
#         name = loc["display_name"]
#         bbox = [float(loc["boundingbox"][2]), float(loc["boundingbox"][0]),
#         float(loc["boundingbox"][3]), float(loc["boundingbox"][1])]
#         with st.expander(name):
#             if st.button(f"Analyze {name}", key=f"{name}-{idx}"):
#                 st.session_state["selected_bbox"] = buffer_bbox(bbox)
#                 st.write(f"📍 Original bbox: {bbox}")
#                 buffered_bbox = buffer_bbox(bbox)
#                 st.write(f"🧭 Buffered bbox: {buffered_bbox}")
#                 with st.spinner("Running analysis..."):
#                     results, results_images = analyze_aoi(buffered_bbox)
#                     if not results.empty:
#                         st.write("### Land Cover Class Trends Over Time")
#                         plot_land_cover_trends(results)
#                         st.write("###Land Cover Maps Over 10 Years")
#                         visualize_multiple_years(results_images, VIS_CLASS_IDS, list(DW_CLASSES.values()), VIS_PALETTE)
#                         # Add map overlay
#                         st.write("###Landcover Map over Satellite Imagery (2025)")
#                         # Show 2025 clipped image if available
#                         tif_path = os.path.join(RASTER_FOLDER, "dw_up_2025.tif")
#                         if os.path.exists(tif_path):
#                             bbox_geom = box(*buffered_bbox)
#                             geojson_geom = [mapping(bbox_geom)]
#                             with rasterio.open(tif_path) as src:
#                                 out_image, _ = mask(src, geojson_geom, crop=True)
#                                 show_overlay_on_map(out_image[0], buffered_bbox)

#                         # Optional: change detection
#                         if os.path.exists(os.path.join(RASTER_FOLDER, 'dw_up_2016.tif')) and \
#                            os.path.exists(os.path.join(RASTER_FOLDER, 'dw_up_2025.tif')):
#                             display_change_detection(buffered_bbox, RASTER_FOLDER, list(DW_CLASSES.values()))

#                         st.success("Done!")
#                     else:
#                         st.warning("No data found for this region.")

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
import tempfile
import json
from utils import buffer_bbox
import urllib.request

RASTER_FOLDER = "/home/azureuser/app/raster_data"

def get_raster_path(year):
    local_path = os.path.join(RASTER_FOLDER, f"dw_up_{year}.tif")
    if not os.path.exists(local_path):
        st.warning(f"Raster file not found: {local_path}")
        return None
    return local_path

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
        # path = os.path.join(RASTER_FOLDER, f"dw_up_{year}.tif")
        path = get_raster_path(year)
        st.write(f"{path}")
        if not os.path.exists(path):
            return "Error in path formation"

        with rasterio.open(path) as src:
            out_image, _ = mask(src, geojson_geom, crop=True)
            data = out_image[0]
            if src.nodata is not None:
                data = data[data != src.nodata]

            unique, counts = np.unique(data, return_counts=True)
            total = counts.sum()
            records[year] = {DW_CLASSES.get(int(cls), str(cls)): round(cnt/total*100, 2) for cls, cnt in zip(unique, counts) if cls in DW_CLASSES}
            records[year]["Year"] = year
            images_by_year[year] = out_image[0]

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
    m.add_geojson(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}, "properties": {"name": "AOI"}}]}), layer_name="AOI")
    m.add_legend(title="Land Cover Classes", legend_dict={DW_CLASSES[i]: VIS_PALETTE[i] for i in range(9)})
    m.to_streamlit(height=550)

def display_change_detection(bbox):
    st.write("## 🔄 Change Detection (2016 → 2025)")
    geojson_geom = [mapping(box(*bbox))]

    def get_array(year):
        path = get_raster_path(year)
        # with rasterio.open(os.path.join(RASTER_FOLDER, f"dw_up_{year}.tif")) as src:
        with rasterio.open(path) as src:
            return mask(src, geojson_geom, crop=True)[0][0]

    try:
        arr_2016, arr_2025 = get_array(2016), get_array(2025)
        if arr_2016.shape != arr_2025.shape:
            st.warning("Masked arrays differ in shape.")
            return
    except Exception as e:
        st.warning(f"Error: {e}")
        return

    # Move dropdowns above the plots to keep layout aligned
    col1, col2 = st.columns(2)
    with col1:
        from_class = st.selectbox("From Class", list(DW_CLASSES.values()), index=1, key="from_class")
    with col2:
        to_class = st.selectbox("To Class", list(DW_CLASSES.values()), index=6, key="to_class")

    from_id = list(DW_CLASSES.keys())[list(DW_CLASSES.values()).index(from_class)]
    to_id = list(DW_CLASSES.keys())[list(DW_CLASSES.values()).index(to_class)]
    mask_change = (arr_2016 == from_id) & (arr_2025 == to_id)

    # Aligned side-by-side plots
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.imshow(arr_2016 != arr_2025, cmap='gray')
        ax1.set_title("Changed Pixels", fontsize=14)
        ax1.axis('off')
        st.pyplot(fig1)

    with plot_col2:
        fig2, ax2 = plt.subplots(figsize=(3, 3))
        ax2.imshow(mask_change, cmap='Reds')
        ax2.set_title(f"{from_class} → {to_class}", fontsize=14)
        ax2.axis('off')
        st.pyplot(fig2)

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
    with st.spinner("Running analysis..."):
        df, images = analyze_aoi(buffered)
        if not df.empty:
            st.write("### Land Cover Class Trends Over Time")
            plot_land_cover_trends(df)
            st.write("### Land Cover Maps Over 10 Years")
            visualize_multiple_years(images)

            tif_path = os.path.join(RASTER_FOLDER, "dw_up_2025.tif")
            if os.path.exists(tif_path):
                with rasterio.open(tif_path) as src:
                    out_image, _ = mask(src, [mapping(box(*buffered))], crop=True)
                    st.write("### Landcover Map over Satellite Imagery (2025)")
                    show_overlay_on_map(out_image[0], buffered)

            with st.expander("Change Detection", expanded=True):
                display_change_detection(buffered)
        else:
            st.warning("No data found for this region.")
