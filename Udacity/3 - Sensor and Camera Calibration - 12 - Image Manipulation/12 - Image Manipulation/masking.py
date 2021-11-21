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
    width, height = img.size
    print(width, height)
    mask = np.zeros((height, width))
    for column in range(width):
        for row in range(height):
            RGB = img.getpixel((column, row))
            if RGB[0] < color_threshold[0] or RGB[1] < color_threshold[1] or RGB[2] < color_threshold[2]:
                mask[row][column] = 0
            else:
                mask[row][column] = 255                          
    
    return img, mask

def mask_and_display(img, mask):
    """
    display 3 plots next to each other: image, mask and masked image
    args:
    - img [array]: HxWxC image array
    - mask [array]: HxW mask array
    """
       
    mask_image = Image.fromarray(mask)
    masked_image2 = mask_image.copy()
    masked_image2.convert("RGB")
    masked_image2 = img * masked_image2
    img2 = img.copy()
    #img2.convert("RGBA")
    #img2.putalpha(mask_image2)
       
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img)
    ax2.imshow(mask_image) 
    ax3.imshow(masked_image2)
    plt.show()
        
    return

if __name__ == '__main__':
    path = 'data/images/segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    color_threshold = [128, 128, 128]
    img, mask = create_mask(path, color_threshold)
    mask_and_display(img, mask)