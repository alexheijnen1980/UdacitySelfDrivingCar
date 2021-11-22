import glob
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sn
from PIL import Image
from PIL import ImageStat


def calculate_mean_std(image_list):
    """
    calculate mean and std of image list
    args:
    - image_list [list[str]]: list of image paths
    returns:
    - mean [array]: 1x3 array of float, channel wise mean
    - std [array]: 1x3 array of float, channel wise std
    """
    means = []
    stds = []
    for path in image_list:
        img = Image.open(path)
        img = img.convert('RGB')
        stat = ImageStat.Stat(img)
        means.append(np.array(stat.mean))
        stds.append(np.array(stat.var)**0.5)
    
    total_mean = np.mean(means, axis = 0) 
    total_std = np.mean(stds, axis = 0)
    
    return total_mean, total_std


def channel_histogram(image_list):
    """
    calculate channel wise pixel value
    args:
    - image_list [list[str]]: list of image paths
    """
    red = []
    green = []
    blue = []  
    for path in image_list:
        img = Image.open(path)
        img = img.convert('RGB')
        img = np.array(img)
        R = img[..., 0] 
        G = img[..., 1]
        B = img[..., 2]
        red.extend(R.flatten().tolist())
        blue.extend(B.flatten().tolist())
        green.extend(G.flatten().tolist())
    # Plot histograms using matplotlib as seaborn is not yet installed.
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.hist(red, color = 'red')
    ax2.hist(green, color = 'green') 
    ax3.hist(blue, color = 'blue')
    plt.show()
                        
if __name__ == "__main__": 
    image_list = glob.glob('data/images/*')
    mean, std = calculate_mean_std(image_list)