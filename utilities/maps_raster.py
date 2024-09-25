"""
Make raster maps.
"""
import numpy as np
import rasterio
from rasterio import features


def make_raster_from_vectors(
        geometry,
        vals_for_colours,
        height,
        width,
        transform
        ):
    """
    # Burn geometries for left-hand map:

    """
    shapes = ((geom, value) for geom, value in zip(geometry, vals_for_colours))
    burned = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        fill=np.NaN,
        transform=transform,
        all_touched=True
    )
    burned = np.flip(burned, axis=0)
    return burned


def set_up_raster_transform(gdf, pixel_size=1000):
    """
    # Code source for conversion to raster:
    # https://gis.stackexchange.com/a/475845
    """
    # Prepare some variables
    xmin, ymin, xmax, ymax = gdf.total_bounds
    width = int(np.ceil((pixel_size + xmax - xmin) // pixel_size))
    height = int(np.ceil((pixel_size + ymax - ymin) // pixel_size))
    transform = rasterio.transform.from_origin(
        xmin, ymax, pixel_size, pixel_size)

    # Store the parameters for scaling the raster image
    # to match the original vector image:
    transform_dict = {
        # For the vector image:
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
        # For the raster image:
        'pixel_size': pixel_size,
        'width': width,
        'height': height,
        'im_xmax': xmin + (pixel_size * width),
        'im_ymax': ymin + (pixel_size * height),
        'transform': transform
    }
    return transform_dict
