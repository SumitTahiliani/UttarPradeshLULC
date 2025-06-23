def buffer_bbox(bbox, lat_buffer=0.00225, lon_buffer=0.0025):
    minx, miny, maxx, maxy = map(float, bbox)
    return [
        minx - lon_buffer,
        miny - lat_buffer,
        maxx + lon_buffer,
        maxy + lat_buffer
    ]