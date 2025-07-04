# Multi-State Land Use Land Cover (LULC) Analysis Tool

A comprehensive web application for analyzing Land Use and Land Cover changes across multiple Indian states using Google's Dynamic World dataset.

---

## Overview

This web app allows you to explore **Land Use and Land Cover (LULC)** changes in **Uttar Pradesh, Maharashtra, and West Bengal, India**, from **2016 to 2025**. It uses **Google's Dynamic World dataset**, clipped to state boundaries, and provides interactive GIS tools for analysis and visualization.

---

## Features

- **Multi-State Support**: Analyze locations in Uttar Pradesh, Maharashtra, and West Bengal
- **Location Search**: Find any place in supported states using the search bar (powered by Nominatim/OpenStreetMap)
- **Land Cover Maps**: View annual land cover maps (2016–2025) for your selected area
- **Trends & Statistics**: See line charts showing the percentage of each land cover class over time
- **Change Detection**: Analyze how much land changed from one class to another between 2016 and 2025
- **Satellite Overlay**: Optionally overlay the land cover map on satellite imagery
- **Download Capabilities**: Download plots and maps as PNG files
- **New Analysis**: Easy reset functionality to analyze multiple locations

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd UttarPradeshLULC
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare raster data**
   - Create the following folder structure in your project root:
     ```
     dw_up_rasters/     # Uttar Pradesh rasters
     dw_mh_rasters/     # Maharashtra rasters  
     dw_wb_rasters/     # West Bengal rasters
     ```
   - Place your raster files in the respective folders with the naming convention:
     - Uttar Pradesh: `dw_up_2016.tif`, `dw_up_2017.tif`, etc.
     - Maharashtra: `dw_mh_2016.tif`, `dw_mh_2017.tif`, etc.
     - West Bengal: `dw_westbengal_2016.tif`, `dw_westbengal_2017.tif`, etc.

4. **Run the application**
   ```bash
   # For local rasters
   streamlit run app.py
   
   # For HuggingFace rasters (downloads automatically)
   streamlit run app_hf.py
   ```

5. **Access the app**
   - Open your browser and go to `http://localhost:8501`
   - The app will automatically detect your location's state and load the appropriate rasters

---

## How to Use

1. **Search for a location** in any supported state using the search bar
2. **Select a suggestion** from the dropdown to analyze that area
3. **Explore the results** using the checkboxes:
   - **Land Cover Maps**: View yearly land cover changes
   - **Trends**: See percentage changes over time
   - **Satellite Overlay**: Overlay land cover on satellite imagery
   - **Change Detection**: Analyze specific land cover transitions
4. **Download results** using the download buttons
5. **Start a new analysis** using the "New Analysis" button

---

## Supported States

- **Uttar Pradesh** (`up`): Full state coverage
- **Maharashtra** (`mh`): Full state coverage (2025 uses 2024 data due to raster limitations)
- **West Bengal** (`wb`): Full state coverage

---

## Data Sources

- **Land Cover Data**: Google Dynamic World dataset (2016-2025)
- **Geocoding**: Nominatim/OpenStreetMap
- **Satellite Imagery**: Leafmap basemaps
- **State Boundaries**: Approximate bounding boxes for state detection

---

## Technical Notes

- **Coordinate System**: All data uses EPSG:4326 (WGS84)
- **Raster Processing**: Automatic downsampling and masking for performance
- **Caching**: Streamlit caching for improved performance
- **Error Handling**: Graceful fallbacks for data issues (e.g., Maharashtra 2025)

---

## Troubleshooting

- **"Location not in a supported state"**: Ensure your search location is within Uttar Pradesh, Maharashtra, or West Bengal
- **Raster loading errors**: Check that raster files are in the correct folders with proper naming
- **Performance issues**: The app automatically downsamples large rasters for better performance
