import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
"""
TODO:
1. Make copy method for each object (AND a copy_from, stolen from /home/ocean/Others/unfoldinggroup/unfolding/unfoldingsuite/datahandler.py)
2. Make model for each
3. Make model fitness param for each
Steps for processing each image:
- Remove horizontal streaks (by performing fourier analysis in the vertical direction)
- Set background noise level
- Extract ALL curvilinear parts, by various techniques, e.g.:
    - use the peak prominence algorithm on each channel, but adapted into 2D (time x strip #)
    - but reframed to something more complicated like:
        - determine the connected component for each level (ignoring spurious holes and dots smaller than a certain size)
        - Pick the ridge of each peak as the trail.
        - Somehow make sure a new trail is started when there's
            - a sharp kink in trail direction
            - a sharp kink in the d(trail intensity)/d(trail length) (i.e. change of slope of intensity of the trail)
    - triangulate the trail at each timepoint, allowing for small offsets
4. Plot the trial in 3D
5. Plot trail ridge on top
6. Plot trail's boundary (i.e. the contour line for noise level +1 ) around the 3 axes
"""

class Event():
    def __init__(self, filename):
        self.path = filename
        data = read_raw_pixel_values(self.path)
        self.u, self.v, self.w = data.transpose([2,0,1]) # each channel has [time, strip #]
        self.model = []
    def model_fitness():
        pass

    @classmethod
    def copy_from(cls):
        copy_of_self = cls.__new__()

def read_raw_pixel_values(filename):
    """Get the raw RBG values"""
    return np.array(Image.open(filename, 'r'))

#weirdly, there are only 21 allowed pixel brightness values.

if __name__=="__main__":
    for num_prongs in range(1,4):
        if num_prongs!=3:
            continue
        # for i in tqdm(range(0,371), desc="Scanning {}-prong events...".format(num_prongs)):
        for i in range(0,371):
            if i!=247:
                continue
            event_name = "train/merged_{}p/evt{}.png".format(num_prongs, i)
            event = Event(event_name)
            break
        break
    # see all pixel values
