import rasterio as rio
import numpy as np
import geopandas as gpd
from shapely import geometry
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import box

# import matplotlib
# import matplotlib.pyplot as plt
import rasterio.mask
from rasterio.mask import mask
# from geocube.api.core import make_geocube
from rasterio.plot import reshape_as_image
from rasterio.features import rasterize

import os
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union
from shapely.ops import unary_union

from dynamic_unet import dynamic_model, jacard



from tensorflow import keras
import segmentation_models as sm



import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import segmentation_models as sm
from keras.utils import normalize





#divide image into more manageable "sub images" off the bat. these are not the patches used for classification 
#splits image into nonoverlapping cells of size NxN pixels
#set N to multiple of desired patch size to maximize extracted patches (1024=128*8)
N=1024

#set M such that patch size is MxM pixels 
M=128

#set path to image
image_path='/Users/weh3/Projects/Data/july22.tif'

cwd_path = os.getcwd()

subims_outpath= cwd_path + '/subims'
patches_outpath= cwd_path + '/patches'


def splitImageIntoCells(img, filename, squareDim, start=0):


    numberOfCellsWide = img.shape[1] // squareDim
    numberOfCellsHigh = img.shape[0] // squareDim
    print(numberOfCellsWide)
    print(numberOfCellsHigh)
    x, y = 0, 0




    count = 0
    count+=start
    for hc in range(numberOfCellsHigh):
        y = hc * squareDim
        for wc in range(numberOfCellsWide):
            x = wc * squareDim
            geom = getTileGeom(img.transform, x, y, squareDim)

            #use this geom to also clip and save intersection with gdb. in this case



            count = count + getCellFromGeom(img, geom, filename, count)

# Generate a bounding box from the pixel-wise coordinates using the original datasets transform property
def getTileGeom(transform, x, y, squareDim):
    corner1 = (x, y) * transform #upper left
    corner2 = (x + squareDim, y + squareDim) * transform
    return geometry.box(corner1[0], corner1[1],
                        corner2[0], corner2[1])

# Crop the dataset using the generated box and write it out as a GeoTIFF
def getCellFromGeom(img, geom, filename, count):
    crop, cropTransform = mask(img, [geom], crop=True)
    created=0
    if(np.min(crop)>0):
        created=1
        writeImageAsGeoTIFF(crop,
                            cropTransform,
                            img.meta,
                            img.crs,
                            filename+"_"+str(count))

    return created


# Write the passed in dataset as a GeoTIFF
def writeImageAsGeoTIFF(img, transform, metadata, crs, filename):
    metadata.update({"driver":"GTiff",
                     "height":img.shape[1],
                     "width":img.shape[2],
                     "transform": transform,
                     "crs": crs})








    with rasterio.open(filename+".tif", "w", **metadata) as dest:
        dest.write(img)


    """Function that generates a binary mask from a vector file (shp or geojson)

    raster_path = path to the .tif;

    shape_path = path to the shapefile or GeoJson.

    output_path = Path to save the binary mask.

    file_name = Name of the file.

    """

    # load raster

    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta
        raster_crs=src.crs

    # load o shapefile ou GeoJson
    train_df = df.explode(ignore_index=True)

    # Verify crs
    if train_df.crs != src.crs:
        print(" Raster crs : {}, Vector crs : {}.\n Convert vector and raster to the same CRS.".format(src.crs,
                                                                                                       train_df.crs))

    # Function that generates the mask
    def poly_from_utm(polygon, transform):
        poly_pts = []

        poly = unary_union(polygon)
        for i in np.array(poly.exterior.coords):
            poly_pts.append(~transform * tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly

    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)


    #print(gpd.GeoSeries(poly_from_utm(gpd.GeoSeries(poly_shp),src.meta['transform'])).to_json())

    # #test
    # testdf=gpd.GeoDataFrame({'geometry': poly_shp, 'data':iter(range(len(poly_shp)))})
    # testdf.plot()
    # print(box(*(testdf.total_bounds)).area)
    # print(unary_union(poly_shp).centroid)
    # test2=unary_union(poly_shp)
    # test3=gpd.GeoDataFrame({'geometry': test2, 'data':iter(range(len(test2)))})
    # test3.crs=raster_crs
    # test3=test3.to_crs('EPSG:4326')
    # print(test3.centroid)
    # #print(testdf.explode(ignore_index=True).dissolve().centroid)
    #
    # plt.show()



    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size)

    # Save
    mask = mask.astype("uint16")

    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    os.chdir(output_path)
    with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
        dst.write(mask * 255, 1)

def splitSubimIntoCells(img, filename, squareDim, counter):
    numberOfCellsWide = img.shape[1] // squareDim
    numberOfCellsHigh = img.shape[0] // squareDim
    x, y = 0, 0

    for hc in range(numberOfCellsHigh):
        y = hc * squareDim
        for wc in range(numberOfCellsWide):
            x = wc * squareDim
            geom = getPatchTileGeom(img.transform, x, y, squareDim)

            created = getPatchCellFromGeom(img, geom, filename, counter)
            if (created == True):
                counter = counter + 1
    return counter

# Generate a bounding box from the pixel-wise coordinates using the original datasets transform property
def getPatchTileGeom(transform, x, y, squareDim):
    corner1 = (x, y) * transform  # upper left
    corner2 = (x + squareDim, y + squareDim) * transform
    return geometry.box(corner1[0], corner1[1],
                        corner2[0], corner2[1])

# Crop the dataset using the generated box and write it out as a GeoTIFF
def getPatchCellFromGeom(img, geom, filename, counter):
    crop, cropTransform = mask(img, [geom], crop=True)


    created = False  
    if(np.percentile(crop, 50)>10):
        writePatchAsGeoTIFF(crop, cropTransform, img.meta, img.crs, filename + '/patch_' + str((counter)))
        writePatchAsGeoTIFF(crop, cropTransform, img.meta, img.crs, filename + '_classified/patch_'  + str((counter)))

        created = True    
    return created

# Write the passed in dataset as a GeoTIFF
def writePatchAsGeoTIFF(img, transform, metadata, crs, filename):
    metadata.update({"driver": "GTiff",
                     "height": img.shape[1],
                     "width": img.shape[2],
                     "transform": transform,
                     "crs": crs})
    with rasterio.open(filename + ".tif", "w", **metadata) as dest:
        dest.write(img)










# Main code begins
with rio.open(image_path) as dataset:
    print(dataset.read(1).shape)

    # compo=np.dstack((dataset.read(4)))


    # plt.imshow(dataset.read(4))
    # plt.show()


    #
    # to start from n+1
    # check dirsize

    _, _, files = next(os.walk(subims_outpath))
    file_count = len(files)



    splitImageIntoCells(dataset, subims_outpath+"/subim", N, start=file_count)






#subims have been saved to /subims folder (subims_outpath) 
















#patch making code


count=0

_, _, files = next(os.walk(subims_outpath))
subim_count = len(files)

 #splits subims into nonoverlapping cells of size MxM pixels
for i in range(subim_count):


    pathname=subims_outpath+'/subim_%s.tif' %i 


#"/Users/william/PycharmProjects/TCN/july22_p/patch" is what it wwas

    with rio.open(pathname) as dataset:
        count=splitSubimIntoCells(dataset, patches_outpath, M, count)





















#unet predictions

_, _, files = next(os.walk(patches_outpath))
n_patches = len(files)

n_classes = 3

metrics = ['accuracy', jacard]

# get images
X_test = []
for i in range(n_patches):
    with rio.open(
            patches_outpath + '/patch_%s.tif' %(i)) as data:
            X_test.append(np.asarray([data.read(1), data.read(2), data.read(3), data.read(4),data.read(5),data.read(6),data.read(7),data.read(8)]))

X_test = np.asarray(X_test)[:, :, :, :]


# use all bands


# train_images2 = train_images.swapaxes(1, 3)
# train_images2 = train_images2.swapaxes(1, 2)

X_test = X_test.swapaxes(1, 3)
X_test = X_test.swapaxes(1, 2)




X_test = normalize(X_test, axis=1)




##test
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


focal_loss = sm.losses.CategoricalFocalLoss()


wce = sm.losses.CategoricalCELoss()
focal_coeff = 2 #trial.suggest_int("focal coeff", 0, 4)
wce_coeff = 0 #trial.suggest_int("wce coeff", 0, 4)
total_loss = focal_coeff * focal_loss + wce_coeff * wce


def dice_coef(y_true, y_pred, smooth=10):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

patch_size = X_test.shape[1]
im_channels = X_test.shape[3]

#optimized filter sizes, batch normalization, and dropout values found through hyperparameter optimization

#filter sizes at each layer
f1 = 3 
f2 = 3 
f3 = 3 
f4 = 3
f5 = 9 
f6 = 3
f7 = 3
f8 = 5 
f9 = 9 

#dropout
d1 = 0.1
d2 = 0.0 
d3 = 0.0
d4 = 0.0
d5 = 0.1

#batch normalization
bn1 = 1
bn2 = 1
bn3 = 1
bn4 = 1
bn5 = 1
bn6 = 0
bn7 = 0
bn8=0 
bn9 = 0

def get_model():
    return dynamic_model(n_classes=n_classes, patch_size=patch_size, num_bands=im_channels,
                       filt1=f1, filt2=f2, filt3=f3, filt4=f4, filt5=f5, filt6=f6, filt7=f7, filt8=f8, filt9=f9,
                       drop1=d1, drop2=d2, drop3=d3, drop4=d4, drop5=d5,
                       norm1=bn1, norm2=bn2, norm3=bn3, norm4=bn4, norm5=bn5, norm6=bn6, norm7=bn7, norm8=bn8,
                       norm9=bn9)

model = get_model()


#learning rate also from hyperparameter optimzation, not relevant to predictions
learn_r = float(0.0007371874448521401) 


callbacks = [EarlyStopping(monitor='val_accuracy', mode='max', patience=50, min_delta=0.02,
                           verbose=1, restore_best_weights=True, start_from_epoch=100)]
opti = "Adam"


if (str(opti) == "Adam"):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_r), loss=total_loss, metrics=metrics)
if (str(opti) == "RMSprop"):
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learn_r), loss=total_loss, metrics=metrics)
if (str(opti) == "SGD"):
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learn_r), loss=total_loss, metrics=metrics)




weights='model_weights_trained.hdf5'

model.load_weights(weights)


for k in len(X_test):
    pred=model.predict(X_test[k])
    with rio.open(
            patches_outpath + '_classified/patch_%s.tif' %(k), "w") as image:
            image.write (pred, 1)
            image.write (np.zeros(np.shape(pred)),  [2,3,4,5,6,7,8])
            

