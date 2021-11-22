import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_mask(path, color_threshold):
    """
    create a binary mask of an image using a color threshold
    args:
    - path [str]: path to image file
    - color_threshold [array]: 1x3 array of RGB value
    returns:
    - img [array]: RGB image array
    - mask [array]: binary array
    """
    img = Image.open(path)
    img = img.convert('RGB')
    img = np.array(img)
    R = img[..., 0] 
    G = img[..., 1]
    B = img[..., 2]
    rt, gt, bt = color_threshold
    # Generate a numpy array with boolean values.
    mask = (R > rt) & (G > gt) & (B > bt)                 
    
    return img, mask

def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
    # Combine the axis with RGB values and the boolean values in the mask.
    masked_image = img * np.stack([mask] * 3, axis = 2)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img)
    ax2.imshow(mask) 
    ax3.imshow(masked_image)
    plt.show()  
    
if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)