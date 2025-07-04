"""
Plotting and visualization functions for Uttar Pradesh LULC project.
"""
import matplotlib.pyplot as plt
import io
import streamlit as st
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Dict
import tempfile
import rasterio
from rasterio.transform import from_bounds
import json
import leafmap.foliumap as leafmap

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

def plot_land_cover_trends(df) -> None:
    """Plot land cover class percentage trends over time."""
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

def visualize_multiple_years(images_by_year: Dict[int, np.ndarray]) -> None:
    """Visualize land cover maps for multiple years."""
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

def show_overlay_on_map(image_array: np.ndarray, bbox) -> None:
    """Show a land cover overlay on a satellite map using leafmap."""
    minx, miny, maxx, maxy = bbox
    height, width = image_array.shape
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    rgb = (ListedColormap(VIS_PALETTE)(image_array / 8.0)[:, :, :3] * 255).astype(np.uint8)
    rgb = np.transpose(rgb, (2, 0, 1))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_tif:
        with rasterio.open(temp_tif.name, "w", driver="GTiff", height=height, width=width, count=3, dtype=rgb.dtype, crs="EPSG:4326", transform=transform) as dst:
            for i in range(3):
                dst.write(rgb[i], i+1)
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