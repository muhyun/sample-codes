import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mxnet import gluon, nd, image
import numpy as np

def check_bbox_in_patch(bbox, patch):

    bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox
    x0, y0, x1, y1 = patch
    included_cnt = 0

    if x0 <= bbox_x0 <= x1 and y0 <= bbox_y0 <= y1:
        included_cnt += 1

    if x0 <= bbox_x1 <= x1 and y0 <= bbox_y1 <= y1:
        included_cnt += 1

    return included_cnt

def get_patch(raw_image, patch_height=224, patch_width=224, bbox=None, bbox_included=False, debug=False, ext=False):

    raw_image_height = raw_image.shape[0]
    raw_image_width = raw_image.shape[1]
    
    if ext:
        org_image_height = raw_image_height // 3
        if debug:
            print(f'org_image_height: {org_image_height}')

#    if debug:
#        print(f'raw_image : {raw_image.shape}, type: {raw_image.dtype}')  
#        print(f'raw height: {raw_image_height}')
#     assert raw_image_height >= patch_height
#     assert raw_image_width >= patch_width

    if raw_image_height < patch_height or raw_image_width < patch_width:
        if debug:
            print('DEBUG - the image is smaller than the desired patch')
        raise Exception('the image is smaller than the desired patch')
        
    if bbox is not None and bbox_included == True:
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox
                
        if ext:
            bbox_y0 += org_image_height
            bbox_y1 += org_image_height
        
#         lower_x0 = max(0, bbox_x1 - patch_width)
#         upper_x0 = min(bbox_x0, raw_image_width - patch_width)

#         lower_y0 = max(0, bbox_y1 - patch_height)
#         upper_y0 = min(bbox_y0, 1 + raw_image_height - patch_height)

        bbox_width = bbox_x1 - bbox_x0
        bbox_height = bbox_y1 - bbox_y0
        
        # If ROI is larger than patch, patch covers 50% or more of the ROI
        if bbox_width > patch_width:
            lower_x0 = max(0, bbox_x0 - int(patch_width/2))
            upper_x0 = min(bbox_x1 - int(patch_width/2), raw_image_width - patch_width)
        else:
            lower_x0 = max(0, bbox_x1 - patch_width)
            upper_x0 = min(bbox_x0, raw_image_width - patch_width)

        if bbox_height > patch_height:
            lower_y0 = max(0, bbox_y0 - int(patch_height/2))
            upper_y0 = min(bbox_x1 - int(patch_height/2), raw_image_height - patch_height)
        else:
            lower_y0 = max(0, bbox_y1 - patch_height)
            upper_y0 = min(bbox_y0, 1 + raw_image_height - patch_height)
            
        if debug:
            print(f'raw_image_height x raw_image_width) - {raw_image_height} x {raw_image_width}')
            print(f'bbox x0, y0, x1, y1 - {bbox}')
            print(f'random x - {lower_x0} ~ {upper_x0}, random y - {lower_y0} ~ {upper_y0}')

        try:
            x0 = np.random.randint(lower_x0, upper_x0)
            y0 = np.random.randint(lower_y0, upper_y0)
            x1 = x0 + patch_width
            y1 = y0 + patch_height
        except:
            if debug:
                print(f'bbox - {bbox} , x0 from {lower_x0} ~ {upper_x0}, y0 from {lower_y0} ~ {upper_y0}')
                print('DEBUG - rand!')
            raise Exception('rand!')

        patch = (x0, y0, x1, y1)
        if debug:
            print(f'patch - {patch}')

        new_bbox_x0 = max(0, bbox_x0 - x0)
        new_bbox_y0 = max(0, bbox_y0 - y0)
        new_bbox_x1 = min(patch_width, bbox_x1 - x0 if bbox_x1 - x0 < patch_width else patch_width)
        new_bbox_y1 = min(patch_height, bbox_y1 - y0 if bbox_y1 - y0 < patch_height else patch_height)
        
        if ext:
            new_bbox_y0 -= org_image_height
            new_bbox_y1 -= org_image_height
        
        new_bbox = (new_bbox_x0, new_bbox_y0, new_bbox_x1, new_bbox_y1)
    else:
        while True:
            x0 = np.random.randint(0, raw_image_width - patch_width + 1) 
            y0 = np.random.randint(0, raw_image_height - patch_height + 1)
            x1 = x0 + patch_width
            y1 = y0 + patch_height

            patch = (x0, y0, x1, y1)
            if bbox is not None:
                bbox_overlapped = check_bbox_in_patch(bbox, patch)
                if bbox_overlapped == 0:
                    break
            else:
                break
                
            if debug:
                print(f'DEBUG - overlapped detected with {bbox_overlapped}')

        new_bbox = (0, 0, 0, 0)

    patch_img = raw_image[y0:y1, x0:x1, :]
    
#    if debug:
#         print(f'patch : {patch} with shape {patch_img.shape}, type: {patch_img.dtype}')  
        
    return patch_img, new_bbox, x0, y0, new_bbox

def draw_with_bbox(img, bbox, debug=False, ext=False):
    plt.imshow(img.asnumpy())

    if ext:
        org_patch_height = img.shape[0]
        
    if bbox is not None:
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = bbox
        bbox_width = bbox_x1 - bbox_x0 + 1
        bbox_height = bbox_y1 - bbox_y0 + 1
                        
        if ext:
            bbox_y0 += org_patch_height
            bbox_y1 += org_patch_height
        
        if debug:
            print(f'draw_with_bbox_ext() - bbox: ({bbox_x0}, {bbox_y0}, {bbox_x1}, {bbox_y1})')

        ax = plt.gca()
        rect = Rectangle((bbox_x0,bbox_y0),bbox_width,bbox_height,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        
def image_expand_3x_height(src, flipped=False):
    """Expand original image to a 3x larger height version.
    
    if flipped == False:
    ------------------------
    |         img          |
    ------------------------
    |         img          |
    ------------------------
    |         img          |
    ------------------------

    if flipped == True:
    ------------------------
    |vertically flipped img|
    ------------------------
    |         img          |
    ------------------------
    |vertically flipped img|
    ------------------------
    
    Parameters
    ----------
    src : mxnet.nd.NDArray
        The original image with HWC format.

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.

    """
    h, w, c = src.shape
    ratio_y = 3
    
    # make canvas
    dst = nd.empty(shape=(h*3, w, c), dtype='uint8')
    
    if flipped:
        dst[0:h, 0:w, :] = nd.flip(src, axis=0)
        dst[h:2*h, 0:w, :] = src
        dst[2*h:3*h, 0:w, :] = nd.flip(src, axis=0)
    else:    
        dst[0:h, 0:w, :] = src
        dst[h:2*h, 0:w, :] = src
        dst[2*h:3*h, 0:w, :] = src
    
    return dst
