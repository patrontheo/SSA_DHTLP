# `python new_synthetic_generator.py --i "image.fits" --o "final" --n 2`
"""
It is a modification of the synthetic_generator.py file that was created in previous project. You can find the original file on : 
https://github.com/YBouquet/detectsat
"""

# from utils.img_processing import morphological_reconstruction
from utils.mosaic import get_crop, get_raw_image
# from utils.lines import get_points

import numpy as np
import cv2
import os
import math
import operator
import random
import gc
import copy
from PIL import Image

def generator_dhtlp(input_file, n_streaks=1, seed=12345678):
    
    raw_image, _ = get_raw_image(input_file)   # get the raw image
    crop = get_crop(raw_image, 0, 0) 

    tmp_mask = []
    # random.seed(seed)

    y_true = np.zeros(crop.shape)
    
    max_ = np.max(crop)
    min_ = np.min(crop)
    median_ = np.median(crop)


    if max_ != min_ : # white patch
        crop = (crop - min_) / (max_ - min_) # rescaling
        mask = np.full(crop.shape, 0.)

        coord = []
        for l in range(n_streaks): # add n_streaks streaks
            # STREAK PARAMETRIZATION
            s_length = random.randint(300,1500) # chose the length of the streak
            s_width = random.randint(6,20) # chose the width of the streak

            min_intensity = np.min([np.max([110, median_]) + 17, 255]) # get minimum intensity of the streaks for it to be visible
            intensity_ = random.randint(min_intensity,255) # chose the intensity of the streak

            x_where, y_where = random.randint(0, crop.shape[0]-s_length-1), random.randint(0,crop.shape[1]-s_length-1) # chose the position of the streak
            
            # DRAWING
            sat_line = np.zeros((s_length,s_length,3)).astype(np.uint8)

            if l == 0 :
                x1 = 30
                x2 = s_length-30
                y1 = random.randint(30,s_length-30)
                y2 = random.randint(30,s_length-30)
            else :
                y1 = 30
                y2 = s_length-30
                x1 = random.randint(30,s_length-30)
                x2 = random.randint(30,s_length-30)
            
            sat_line = cv2.line(sat_line, (x1, y1), (x2, y2), (intensity_,intensity_,intensity_), s_width)

            blur_sat_line = erosion_(sat_line, s_length) #erode the streak on some small part to make it less uniform
            blur_sat_line = intensity(blur_sat_line, intensity_=intensity_) #change the intensity of the streak on some small part to make it less uniform
            
            final_synth = cv2.GaussianBlur(blur_sat_line/255,(s_width*2-7,s_width*2-7),0)

            alpha_trans = 85/100. # random.randint(75,95)/100. # opacity of the streak

            final_synth = (final_synth / np.max(final_synth))
            tmp_mask.append(final_synth)
            mask[x_where:x_where+s_length, y_where:y_where+s_length] = final_synth
            indices = np.argwhere(mask > 0.)
            for subx, suby in indices :
                crop[subx,suby] = max(alpha_trans * mask[subx,suby] + (1-alpha_trans) * crop[subx,suby], crop[subx,suby])

            y_true[x_where:x_where+s_length, y_where:y_where+s_length] = (sat_line[:,:,0] > intensity_-30).astype(int)
            del sat_line
            del blur_sat_line
            gc.collect()

            coord.append([(x1+y_where, y1+x_where), (x2+y_where, y2+x_where)])
        crop = crop * (max_ - min_) + min_

        sub_max = np.max(crop)
        sub_min = np.min(crop)
        crop = (crop - sub_min) / (sub_max - sub_min)

    return crop, y_true, coord


def erosion_(image, length) : 

    number_alteration = np.random.randint(5,7)
    trace = np.argwhere(image[:,:,0] > 0)

    for i in range(number_alteration):
        index = random.randint(1, trace.shape[0]-1)
        h, w = trace[index][0], trace[index][1]
        size = random.randint(20,40)
        image2 = image[h-size : h+size, w-size : w+size, :].copy()
        
        if image2.shape[0] == 0 or image2.shape[1] == 0: #if the selected point is on the edge, erosion will not be possible so we skip it
            continue                # may be better to check that beforehand

        eroded = cv2.erode(image2/255, np.ones((2,2), dtype="uint8"), iterations=1)*255
        image[h-size : h+size, w-size : w+size, :] = eroded
        
    return image[:,:,0]

def intensity(image, intensity_): 

    number_alteration = np.random.randint(3,6)
    trace = np.argwhere(image > 0)

    for i in range(number_alteration):
        index = random.randint(1, trace.shape[0]-1)
        h, w = trace[index][0], trace[index][1]
        size = random.randint(20,40)
        image2 = image[h-size : h+size, w-size : w+size].copy()
        
        sub_trace = np.argwhere(image2 > 0)
        mu = random.randint(intensity_-10, intensity_-5)
        values = np.random.randint(mu-7,mu+7, sub_trace.shape[0])
        image2[sub_trace[:,0], sub_trace[:,1]] = values

        image[h-size : h+size, w-size : w+size] = image2
        
    return image


########################################################################################################################


# def saturated_stars(unscaled_img):
#     """
#     Get a mask with all saturated light blob by thresholding and morphological reconstruction
#     """
#     sigma = np.std(unscaled_img.flatten())
#     indices = np.argwhere(unscaled_img > 3*sigma)
#     mask = np.zeros(unscaled_img.shape).astype(np.uint8)
#     mask[indices[:,0], indices[:,1]] = 1
#     mask_1 = np.zeros(unscaled_img.shape).astype(np.uint8)
#     indices = np.argwhere( unscaled_img > np.mean(unscaled_img))
#     mask_1[indices[:,0], indices[:,1]] = 1
#     final_mask = morphological_reconstruction(mask, mask_1, 2)
#     return final_mask #, boxes

# def blurry_effect(image, length) : 
#     blurry = np.zeros(shape=(length,length))
#     treshold = np.random.uniform(low = 0.2, high = 0.5)
#     for i2 in range(length) : 
#         for j2 in range(length) : 
#             mean = int(length/2 - 1)
#             x = -mean + j2 
#             y = -mean + i2 
#             dist = np.sqrt(x**2 + y**2) / (np.sqrt(2*mean**2))
#             if image[i2,j2,0] != 0 : 
#                 if dist > treshold : 
#                     new_value = image[i2,j2,0] - ((255-100)*dist)
#                 else :
#                                                     # keep a core without decreasing
#                     new_value = image[i2,j2,0] 
#             else : 
#                 new_value = 0
        
#             blurry[i2,j2] = new_value  
#     return blurry 

# def salt_pepper_noise(img, prob=0.05):
 
#     # Getting the dimensions of the image
#     row , col, _ = img.shape
     
#     # Randomly pick some pixels in the
#     # image for coloring them white
#     # Pick a number of pixels to be colored white (*0.5 because half salt and half pepper)
#     number_of_pixels = int(np.ceil(row * col * 0.5 * prob))
#     for i in range(number_of_pixels):
       
#         # Pick a random y coordinate
#         y_coord=random.randint(0, row - 1)
         
#         # Pick a random x coordinate
#         x_coord=random.randint(0, col - 1)
         
#         # Color that pixel to white
#         img[y_coord][x_coord] = 1.0
         
#     # Randomly pick some pixels in
#     # the image for coloring them black
#     for i in range(number_of_pixels):
       
#         # Pick a random y coordinate
#         y_coord=random.randint(0, row - 1)
         
#         # Pick a random x coordinate
#         x_coord=random.randint(0, col - 1)
         
#         # Color that pixel to black
#         img[y_coord][x_coord] = 0.0
         
#     return img


# def generator(input_file, seed = 12345678, intensity_=255):
#     raw_image, _ = get_raw_image(input_file)#"OMEGA.2020-01-29T03_51_46.345_fullfield_binned.fits")

#     tmp_mask = []

#     # random.seed(seed)

#     crop = get_crop(raw_image, 0, 0)
#     #unscaled_crop = get_crop(unscaled_img, 0, 0)
    
#     y_true = np.zeros(crop.shape)
    
#     #refcrop = crop.copy()
#     #subdilation = final_mask[alpha:alpha+subh, beta:beta+subw]
#     #star_indices = np.argwhere(subdilation == 1)
#     #replacement_value = np.median(subcrop)
#     #subcrop[star_indices[:,0], star_indices[:,1]] = replacement_value # remove outlier pixels
#     max_ = np.max(crop)
#     min_ = np.min(crop)
#     if max_ != min_ : # white patch
#         crop = (crop - min_) / (max_ - min_) # rescaling
#         mask = np.full(crop.shape, 0.)
#         decision = random.random()
#         if decision < 1:    #0.5
#             # STREAK PARAMETRIZATION
#             s_length = random.randint(1000,2000) # chose the length of the streak
#             s_width = random.randint(10,50) # chose the width of the streak
#             theta = random.randint(0,179) * math.pi / 180. # chose the direction of the streak (in radian)
#             x_where, y_where = random.randint(0, crop.shape[0]-s_length-1), random.randint(0,crop.shape[1]-s_length-1) # chose the position of the streak

#             # DRAWING
#             sat_line = np.zeros((s_length,s_length,3)).astype(np.uint8)
#             _,p1,p2 = get_points(0,theta)
#             sat_line = cv2.line(sat_line, tuple(map(operator.add, p1,(int(s_length/2),int(s_length/2)))),tuple(map(operator.add, p2,(int(s_length/2),int(s_length/2)))), (intensity_,intensity_,intensity_), s_width)

#             # APPLY THE STREAK IN THE PATCH
#             h_sat,w_sat,_ = sat_line.shape
#             cx,cy = float(h_sat//2), float(w_sat//2)
#             r = min(cx,cy)
#             for a in range(h_sat):
#                 for b in range(w_sat):
#                     if math.sqrt((a - cx)**2 + (b-cy)**2) > r:
#                         sat_line[a,b] = 0
            
#             blurry_random = np.random.uniform(low = 0, high = 1 )
#             if blurry_random <= 1. : #0.5 : #around 50% of blurry effect                        #TO DELETE?
#                 blur_sat_line = blurry_effect_2(sat_line, s_length)    
#             else : 
#                 blur_sat_line = sat_line[:,:,0]

#             blur_sat_line = intensity(blur_sat_line, intensity_=intensity_) 
            
#                     #final_synth = cv2.GaussianBlur(sat_line/255,(s_width*2-1,s_width*2-1),0)
#             final_synth = cv2.GaussianBlur(blur_sat_line/255,(s_width*2-15,s_width*2-15),0)
#             # final_synth = blur_sat_line/255.
#                     #final_synth = cv2.GaussianBlur(sat_line[:,:,0]/255,(s_width*2-1,s_width*2-1),0)

#             alpha_trans = 85/100. # random.randint(75,95)/100. # opacity of the streak

#             final_synth = (final_synth / np.max(final_synth))
#             tmp_mask.append(final_synth)
#             mask[x_where:x_where+s_length, y_where:y_where+s_length] = final_synth
#             indices = np.argwhere(mask > 0.)
#             for subx, suby in indices :
#                 crop[subx,suby] = max(alpha_trans * mask[subx,suby] + (1-alpha_trans) * crop[subx,suby], crop[subx,suby])
#             #y_true[x_where:x_where+s_length, y_where:y_where+s_length] = (sat_line / 255).astype(int)
#             y_true[x_where:x_where+s_length, y_where:y_where+s_length] = (sat_line[:,:,0] > intensity_-30).astype(int)
#             del sat_line
#             del blur_sat_line
#             gc.collect()
#         crop = crop * (max_ - min_) + min_
#         #subcrop[star_indices[:,0], star_indices[:,1]] = refcrop[star_indices[:,0], star_indices[:,1]] # put the blobs in the image back
#         sub_max = np.max(crop)
#         sub_min = np.min(crop)
#         crop = (crop - sub_min) / (sub_max - sub_min)


#     #final_mask = saturated_stars(unscaled_crop) # detect saturated light blob in the image
#     #mask_dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#     #final_mask=cv2.dilate(final_mask,mask_dil, iterations=1)

#     return crop, y_true

# def patcher(crop, y_true, num_samples, mirror = False, ht_iht = False):

#     x_train = []
#     y_train = []
#     has_satellite = []

#     if y_true is None: #if y_true not provided, we put a blank mask (that won't be used later)
#         y_true = np.ones(crop.shape)

#     h,w = crop.shape
    
#     # subh, subw = 696, 781
#     subh, subw = 688, 768
#     if ht_iht:
#         subh, subw = 2588, 3124
#         # subh, subw = 4176, 3124

#     for k in range(num_samples): # number of samples generated from a single patch
#         for alpha in range(0,h, subh):
#             for beta in range(0,w, subw):
#                 if (alpha + subh) <= h and (beta+subw) <= w :
#                     subcrop = crop[alpha:alpha+subh, beta:beta+subw]
#                     target = y_true[alpha:alpha+subh, beta:beta+subw]
#                     hs = 1 if target.sum() > 200 else 0                 #200 is quite arbitrary here
                    
#                     if mirror:
#                         x_mirror = random.randint(0,1)
#                         y_mirror = random.randint(0,1)
#                     else:
#                         x_mirror = 0
#                         y_mirror = 0

#                     if x_mirror == 1:
#                         subcrop = subcrop[::-1]
#                         target = target[::-1]
#                     if y_mirror == 1:
#                         subcrop = subcrop[:,::-1]
#                         target = target[:,::-1]
                    
#                     #refcrop = subcrop.copy()
#                     #subdilation = final_mask[alpha:alpha+subh, beta:beta+subw]
#                     #star_indices = np.argwhere(subdilation == 1)
#                     #replacement_value = np.median(subcrop)
#                     #subcrop[star_indices[:,0], star_indices[:,1]] = replacement_value # remove outlier pixels
                    

#                     # SAVING THE SAMPLES that contains satellites
#                     if hs:
#                         has_satellite.append(hs)
#                         x_train.append(subcrop)
#                         y_train.append(target)
#                     # elif random.randint(0,15) == 0: # add some patches that do not contain streak
#                     #     has_satellite.append(hs)
#                     #     x_train.append(subcrop)
#                     #     y_train.append(target)

    
#     # _, out_file = os.path.split(input_file)
    
#     # if not os.path.exists(outDir):
#     #     os.makedirs(outDir)
#     # np.save(os.path.join(outDir, outFile) + "_samples.npy", np.array(x_train))
#     # np.save(os.path.join(outDir, outFile) + "_targets.npy", np.array(y_train))
#     # np.save(os.path.join(outDir, outFile) + "_patch_targets.npy", np.array(has_satellite))
    
#     x_train = np.expand_dims(np.array(x_train), 3)
#     y_train = np.expand_dims(np.array(y_train), 3)
#     has_satellite = np.array(has_satellite)

#     return x_train, y_train, has_satellite


# # if __name__ == '__main__':
# #     main(prologue.get_args())

# def loader(input_file):
#     raw_image, _ = get_raw_image(input_file)
#     crop = get_crop(raw_image, 0, 0)
#     sub_max = np.max(crop)
#     sub_min = np.min(crop)
#     crop = (crop - sub_min) / (sub_max - sub_min)
    
#     label = os.path.splitext(input_file)[0]
#     try:
#         img_PIL = Image.open(label + '.jpg').convert('L')
#         y_true = (np.array(img_PIL) > 10).astype(int)
#     except:
#         y_true = None
#     return crop, y_true



# def generator_intensity(input_file, seed = 12345678, intensities=[255]):
#     raw_image, _ = get_raw_image(input_file)#"OMEGA.2020-01-29T03_51_46.345_fullfield_binned.fits")

#     tmp_mask = []

#     # random.seed(seed)

#     crop = get_crop(raw_image, 0, 0)
#     #unscaled_crop = get_crop(unscaled_img, 0, 0)
    
#     y_true = np.zeros(crop.shape)
    
#     #refcrop = crop.copy()
#     #subdilation = final_mask[alpha:alpha+subh, beta:beta+subw]
#     #star_indices = np.argwhere(subdilation == 1)
#     #replacement_value = np.median(subcrop)
#     #subcrop[star_indices[:,0], star_indices[:,1]] = replacement_value # remove outlier pixels
#     max_ = np.max(crop)
#     min_ = np.min(crop)
#     if max_ != min_ : # white patch
#         crop = (crop - min_) / (max_ - min_) # rescaling
#         mask = np.full(crop.shape, 0.)
#         decision = random.random()
#         if decision < 1:    #0.5
#             # STREAK PARAMETRIZATION
#             s_length = random.randint(1000,2000) # chose the length of the streak
#             s_width = random.randint(10,50) # chose the width of the streak
#             theta = random.randint(0,179) * math.pi / 180. # chose the direction of the streak (in radian)
#             x_where, y_where = random.randint(0, crop.shape[0]-s_length-1), random.randint(0,crop.shape[1]-s_length-1) # chose the position of the streak

#             # DRAWING
#             _,p1,p2 = get_points(0,theta)

#             imgs = []
#             annot = []
#             for intensity_ in intensities:
#                 sat_line = np.zeros((s_length,s_length,3)).astype(np.uint8)
#                 crop_tmp = copy.deepcopy(crop)
#                 mask_tmp = copy.deepcopy(mask)
#                 y_true_tmp = copy.deepcopy(y_true)
#                 sat_line = cv2.line(sat_line, tuple(map(operator.add, p1,(int(s_length/2),int(s_length/2)))),tuple(map(operator.add, p2,(int(s_length/2),int(s_length/2)))), (intensity_,intensity_,intensity_), s_width)

#                 # APPLY THE STREAK IN THE PATCH
#                 h_sat,w_sat,_ = sat_line.shape
#                 cx,cy = float(h_sat//2), float(w_sat//2)
#                 r = min(cx,cy)
#                 for a in range(h_sat):
#                     for b in range(w_sat):
#                         if math.sqrt((a - cx)**2 + (b-cy)**2) > r:
#                             sat_line[a,b] = 0
                
#             # blurry_random = np.random.uniform(low = 0, high = 1 )
#             # if blurry_random <= 1. : #0.5 : #around 50% of blurry effect                        #TO DELETE?
#             #     blur_sat_line = blurry_effect_2(sat_line, s_length)    
#             # else : 
#             #     blur_sat_line = sat_line[:,:,0]

#                 # blur_sat_line = intensity(blur_sat_line, intensity_=intensity_) 
            
#                     #final_synth = cv2.GaussianBlur(sat_line/255,(s_width*2-1,s_width*2-1),0)
#                 final_synth = cv2.GaussianBlur(sat_line[:,:,0]/255,(s_width*2-15,s_width*2-15),0)
#                 # final_synth = blur_sat_line/255.
#                         #final_synth = cv2.GaussianBlur(sat_line[:,:,0]/255,(s_width*2-1,s_width*2-1),0)

#                 alpha_trans = 85/100. # random.randint(75,95)/100. # opacity of the streak

#                 # final_synth = (final_synth / np.max(final_synth))

#                 mask_tmp[x_where:x_where+s_length, y_where:y_where+s_length] = final_synth
#                 indices = np.argwhere(mask_tmp > 0.)
#                 for subx, suby in indices :
#                     crop_tmp[subx,suby] = max(alpha_trans * mask_tmp[subx,suby] + (1-alpha_trans) * crop_tmp[subx,suby], crop_tmp[subx,suby])
#                 #y_true[x_where:x_where+s_length, y_where:y_where+s_length] = (sat_line / 255).astype(int)
#                 y_true_tmp[x_where:x_where+s_length, y_where:y_where+s_length] = (sat_line[:,:,0] > intensity_-40).astype(int)

#                 crop_tmp = crop_tmp * (max_ - min_) + min_
#                 #subcrop[star_indices[:,0], star_indices[:,1]] = refcrop[star_indices[:,0], star_indices[:,1]] # put the blobs in the image back
#                 sub_max = np.max(crop_tmp)
#                 sub_min = np.min(crop_tmp)
#                 crop_tmp = (crop_tmp - sub_min) / (sub_max - sub_min)

#                 imgs.append(crop_tmp)
#                 annot.append(y_true_tmp)

#     return imgs, annot




# def patcher_dhtlp(crop, y_true, extremities, num_samples):

#     x_train = []
#     y_train = []
#     has_satellite = []

#     if y_true is None: #if y_true not provided, we put a blank mask (that won't be used later)
#         y_true = np.ones(crop.shape)

#     h,w = crop.shape
    
#     # subh, subw = 696, 781
#     subh, subw = 4176, 3124

#     for k in range(num_samples): # number of samples generated from a single patch
#         for alpha in range(0,h, subh):
#             for beta in range(0,w, subw):
#                 if (alpha + subh) <= h and (beta+subw) <= w :
#                     subcrop = crop[alpha:alpha+subh, beta:beta+subw]
#                     target = y_true[alpha:alpha+subh, beta:beta+subw]
#                     hs = 1 if target.sum() > 200 else 0                 #200 is quite arbitrary here
                    
#                     # SAVING THE SAMPLES that contains satellites
#                     if hs:
#                         has_satellite.append(hs)
#                         x_train.append(subcrop)
#                         y_train.append(target)

#     x_train = np.expand_dims(np.array(x_train), 3)
#     y_train = np.expand_dims(np.array(y_train), 3)
#     has_satellite = np.array(has_satellite)

#     return x_train, y_train, has_satellite
