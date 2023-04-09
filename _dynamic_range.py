import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_raw_pixel_values(filename):
    """Get the raw RBG values"""
    return np.array(Image.open(filename, 'r'))

#weirdly, there are only 21 allowed pixel brightness values.

if __name__=="__main__":
    set_of_all_pixel_values = set()
    for num_prongs in range(1,4):
        for i in tqdm(range(0,371), desc="Scanning {}-prong events...".format(num_prongs)):
            event_name = "train/merged_{}p/evt{}.png".format(num_prongs, i)
            these_pixel_values = np.unique(read_raw_pixel_values(event_name))
            set_of_all_pixel_values = set_of_all_pixel_values.union(set(these_pixel_values))
        
    actual_pixel_values = sorted(list(set_of_all_pixel_values))
    print(actual_pixel_values)