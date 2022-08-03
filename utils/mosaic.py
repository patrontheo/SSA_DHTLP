from typing import Match
from astropy.visualization import ZScaleInterval
from astropy.io import fits

import numpy as np

def get_block(img, x_addresses, y_addresses):
    """
    THIS FUNCTION IS NO LONGER USED IN THE CURRENT PROCESSING SCHEME!

    Parameters
    ----------
    img : list[int]
        full mosaic
    x_addresses : numpy.array(np.float32)
        [min_x, max_x]
    y_addresses : list[int]
        [min_y, max_y]
    Return
    ----------
    result : numpy.array(np.float32)
        corresponding image block
    """
    min_x, max_x = x_addresses
    min_y, max_y = y_addresses
    result = img[1][min_x:max_x, min_y : max_y]
    if np.any(np.isnan(result)):
        raise Exception('Some values are NAN')
    return result

def scale_image(raw_img):
    """
    Rescale pixels intensity according to the iraf's ZScale Algorithm

    Parameters
    ----------
    raw_img : numpy.array(np.float32)
        Mosaic with raw intensities
    Return
    ----------
    raw_img : numpy.array(np.float32)
        Rescaled mosaic
    """
    s = ZScaleInterval()
    z1,z2 = s.get_limits(raw_img)
    raw_img[raw_img > z2] = z2
    raw_img[raw_img < z1] = z1
    return raw_img

def get_raw_image(filename):
    """
    Retrieve the mosaic from the fits file and apply a ZScale on it. The final list
    contains the 32 fits images.

    Parameters
    ----------
    filename : string

    Return
    ----------
    scaled_images, raw_images : (numpy.array(np.float32), numpy.array(np.float32))
        Rescaled mosaic, unscaled mosaic
    """
    hdul = fits.open(filename)
    raw_images = []
    scaled_images = []
    if len(hdul) > 1:
        for i in range(1, len(hdul)):
            raw_images.append(hdul[i].data)
            scaled_images.append(scale_image(raw_images[i-1][::-1].copy()))
    elif len(hdul) == 1:
        raw_images.append(hdul[0].data)
        scaled_images.append(scale_image(raw_images[0][::-1].copy()))
    else:
        raise Exception('Fit file empty')
    hdul.close()
    return scaled_images, raw_images[::-1]
    
def get_crop(img, i, j):
    return img[i * 8 + j]

def get_blocks_addresses(raw_img):
    """
    THIS FUNCTION IS NO LONGER USED IN THE CURRENT PROCESSING SCHEME!

    Isolating the 32 blocks of the mosaic by registering for each block
    the indices of their corners

    Parameters
    ----------
    raw_img : numpy.array(np.float32)
        Zscaled mosaic
    Return
    ----------
    crops_addresses : dict
    """
    nans = np.argwhere(np.isnan(raw_img))
    bunches = nans[(nans[:,1] == 0) | (nans[:,0] == 0)]
    cuts_y = []
    cuts_x = []
    temp_y = -1
    temp_x = -1
    for i in bunches :
        x,y = tuple(i)
        if x == 0 :
            if abs(y - temp_y) > 1:
                cuts_y.append([temp_y+1, y])
            temp_y = y
        if y == 0:
            if abs(x - temp_x) > 1:
                cuts_x.append([temp_x+1, x])
            temp_x = x
    cuts_x.append([temp_x+1, raw_img[1].shape[0]])
    cuts_y.append([temp_y+1, raw_img[1].shape[1]])
    cuts_y = np.array(cuts_y)
    cuts_x = np.array(cuts_x)
    crops_addresses = {}
    current_study_x = cuts_x
    current_study_y = cuts_y
    for t_x in current_study_x:
        min_x, max_x = tuple(t_x)
        list_tx = [tuple(t_x)]
        if max_x - min_x > 7000:
            tmp_middle = int((max_x + min_x) / 2)
            list_tx = [(min_x, tmp_middle), (tmp_middle, max_x)]
        for sub_tx in list_tx:
            crops_addresses[sub_tx] = []
            for t_y in current_study_y:
                 crops_addresses[sub_tx].append(tuple(t_y))

    return crops_addresses

# def map_blocks_adresses(raw, column):
#     """
#     Maps the adresses of the individual blocks from a given raw and column identifier of the 4x8 matrix to the corresponding mosaic identifier (1 to 32) given the following convention.

#     |32|31|30|29|16|15|14|13|
#     |28|27|26|25|12|11|10| 9|
#     |24|23|22|21| 8| 7| 6| 5|
#     |20|19|18|17| 4| 3| 2| 1|

#     Parameters
#     ----------
#     raw : int
#     column : int

#     Return
#     ----------
#     adress of the block in the mosaic given the raw and column identifier
#     """
#     raw_colomn_id = '(%i,%i)' %(raw, column)
#     match raw_colomn_id:
#         case '(3,0)':
#             return 0
#         case '(3,1)':
#             return 1
#         case '(3,2)':
#             return 2
#         case '(3,3)':
#             return 3
#         case '(2,0)':
#             return 4
#         case '(2,1)':
#             return 5
#         case '(2,2)':
#             return 6
#         case '(2,3)':
#             return 7
#         case '(1,0)':
#             return 8
#         case '(1,1)':
#             return 9
#         case '(1,2)':
#             return 10
#         case '(1,3)':
#             return 11
#         case '(0,0)':
#             return 12
#         case '(0,1)':
#             return 13
#         case '(0,2)':
#             return 14
#         case '(0,3)':
#             return 15
#         case '(3,4)':
#             return 16
#         case '(3,5)':
#             return 17
#         case '(3,6)':
#             return 18
#         case '(3,7)':
#             return 19
#         case '(2,4)':
#             return 20
#         case '(2,5)':
#             return 21
#         case '(2,6)':
#             return 22
#         case '(2,7)':
#             return 23
#         case '(1,4)':
#             return 24
#         case '(1,5)':
#             return 25
#         case '(1,6)':
#             return 26
#         case '(1,7)':
#             return 27
#         case '(0,4)':
#             return 28
#         case '(0,5)':
#             return 29
#         case '(0,6)':
#             return 30
#         case '(0,7)':
#             return 31