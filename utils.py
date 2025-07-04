"""
General utility functions for the Uttar Pradesh LULC project.
Only keep small, generic helpers here (e.g., buffer_bbox).
"""

# Approximate bounding boxes for supported states
STATE_BBOXES = {
    "Uttar Pradesh": [77.0, 23.8, 84.9, 30.5],    # [minx, miny, maxx, maxy]
    "Maharashtra":   [72.6, 15.6, 80.9, 22.1],
    "West Bengal":   [85.8, 21.4, 89.9, 27.3],
}

STATE_SHORTHAND = {
    "Uttar Pradesh": "up",
    "Maharashtra": "mh",
    "West Bengal": "wb",
}

def buffer_bbox(bbox, lat_buffer=0.00225, lon_buffer=0.0025):
    minx, miny, maxx, maxy = map(float, bbox)
    return [
        minx - lon_buffer,
        miny - lat_buffer,
        maxx + lon_buffer,
        maxy + lat_buffer
    ]

def detect_state_from_bbox(bbox):
    """
    Detect which state a bounding box belongs to by checking if its centroid
    falls within any of the predefined state bounding boxes.
    
    Args:
        bbox: List of [minx, miny, maxx, maxy] coordinates
        
    Returns:
        tuple: (state_name, state_code) or (None, None) if not found
    """
    minx, miny, maxx, maxy = map(float, bbox)
    centroid_lon = (minx + maxx) / 2
    centroid_lat = (miny + maxy) / 2
    
    for state_name, state_bbox in STATE_BBOXES.items():
        state_minx, state_miny, state_maxx, state_maxy = state_bbox
        if (state_minx <= centroid_lon <= state_maxx and 
            state_miny <= centroid_lat <= state_maxy):
            return state_name, STATE_SHORTHAND[state_name]
    
    return None, None