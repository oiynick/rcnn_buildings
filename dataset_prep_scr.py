import numpy as np
import shapely
from aeronet import dataset as ds


def one_sample(fp_json, fp_tif):

    # Read the JSON file and the TIF image
    av_features = ds.vector.FeatureCollection.read(fp_json)
    raster = ds.raster.Band(fp_tif)

    # Image sizing info and setting up the coordinates
    samples = raster.generate_samples(512, 512)

    for i in samples:
        # Taking bounds of the generated sample
        bounds = i.bounds

        # Create a shape of the polygon
        area = shapely.geometry.polygon((bounds.min_point + bounds.max_point))

        # Create a feature based on a shape
        a_feature = ds.vector.Feature(area)

        # Find the intersecting features
        inter = av_features.intersection(a_feature)

        # For every feature make a feature collection and raster data
        for feature in inter:
            offc = ds.vector.FeatureCollection(feature)
            rasta = ds.transforms.rasterize(offc, i.transform, (512, 512))
            yield rasta


def main():
    amount = 1
    res = np.empty(amount, [['complex']])

    for i in range(amount):
        fp_tif = '{}.tif'.format(i)
        fp_json = '{}.json'.format(i)
        res[i] = one_sample(fp_json, fp_tif)
