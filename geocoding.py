"""
Geocoding and location search utilities for Uttar Pradesh LULC project.
"""
import requests
import streamlit as st
from typing import List

@st.cache_data
def search_location(query: str) -> List[dict]:
    """Search for a location using Nominatim API."""
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