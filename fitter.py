import sys
from PIL import Image
import numpy as np
from numpy import array as ary
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label as connected_comp_label
from scipy.signal import convolve2d
"""
TODO:
Steps for processing each image:
[x]Remove horizontal streaks (by performing fourier analysis in the vertical direction)
[x]Set background noise level
[x]Extract ALL curvilinear parts, by various techniques, e.g.:
    [x]use the peak prominence algorithm on each channel, but adapted into 2D (time x strip #)
    [x]but reframed to something more complicated like:
        - determine the connected component for each level (ignoring spurious holes and dots smaller than a certain size)
        - Pick the ridge of each peak as the trail.
[ ]Get the true outline of each blob
    - THINK OF IT AS A SPANNING TREE!
    - fill out all of the holes first.
    - less-than-fully-connected nodes should be near the edge! remove them?
    - Use the 'wildfire' algorithm!
    [ ]OH that was easy, just convolve and then grep for all of the values !=0 or 1.
[ ]Get the rough backbones of each blob, by burning away the edges.
    [ ]Correct this backbone using circular/gaussian weight-windows to find a series of weighted corrected backbone.
        - Or I can use argmax within that circle. Might be a bit ugly but it'll work.
        [ ]resolution = 0.5 pixel?
[ ]Model it as some sort of distribution?


[ ]Somehow make sure a new trail is started when there's
    - a sharp kink in trail direction
    - a sharp kink in the d(trail intensity)/d(trail length) (i.e. change of slope of intensity of the trail)
[ ]triangulate the trail at each timepoint, allowing for small offsets
4. Plot the trails in 3D
5. Plot trail ridge on top
6. Plot trail's boundary (i.e. the contour line for noise level +1 ) around the 3 axes
1. Make copy method for each object (AND a copy_from, stolen from /home/ocean/Others/unfoldinggroup/unfolding/unfoldingsuite/datahandler.py)
2. Make model for each
3. Make model fitness param for each
"""

#weirdly, there are only 21 allowed pixel brightness values.
TRANSLATION_TABLE = {
    0:0,
    12:1,
    25:2,
    39:3,
    51:4,
    64:5,
    77:6,
    90:7,
    102:8,
    115:9,
    128:10,
    141:11,
    153:12,
    166:13,
    179:14,
    192:15,
    204:16,
    217:17,
    229:18,
    242:19,
    254:20,
}
vector_table = np.vectorize(TRANSLATION_TABLE.get)

def read_raw_pixel_values(filename):
    """Get the raw RBG values"""
    return ary(Image.open(filename, 'r'))

class OperationOrderError(Exception):
    """Operation did not proceed in the required order error"""
    pass

_mnorm = lambda x: x/x.max()

class Event():
    def __init__(self, filename):
        self.path = filename
        data = read_raw_pixel_values(self.path)
        # Turn the R,G,B channels back into u, v, w.
        u, v, w = data.transpose([2,0,1]) # each channel has [time, strip #]
        self.u, self.v, self.w = vector_table(u), vector_table(v), vector_table(w)

    def denoise(self):
        """Remove noise from the image using custom methods"""
        raise DeprecationWarning("Don't use this method as it will only destroy information.")
        for strip_direction in "uvw":
            raw_data = getattr(self, strip_direction)
            row_specific_marker = np.percentile(raw_data, 85)
            cleaned_data = np.where((raw_data.T>row_specific_marker).T, raw_data, 0)
            setattr(self, strip_direction, cleaned_data)

    def label_large_chunks(self, percentile_threshold=97, size_threshold=40):
        """
        Highlights and label any significant features on a single channel.
        Parameters
        ----------
        percentile_threshold: float: range(0, 100)
            If the response reaches above the percentile_threshold for that particular row,
            then this response would be deemed significant enough for further evaluation.
        size_threshold: int
            The threshold below which any continguous of cells highlighted as
            'signficant' would be dismissed as noise. And above this threshold,
            the same blob would be considered as signal instead.

        """
        for strip_direction in tqdm("uvw", desc="Finding strictly-signal in each channel"):
            raw_data = getattr(self, strip_direction)
            row_specific_marker = np.percentile(raw_data, percentile_threshold, axis=1)
            significant_cells_bool = (raw_data.T>row_specific_marker).T
            setattr(self, strip_direction+"_row_threshold_used", row_specific_marker)
            # marker = np.percentile(raw_data.flatten(), percentile_threshold)
            # significant_cells_bool = raw_data>marker

            connected_component_map, num_components = connected_comp_label(significant_cells_bool)
            # initilize empty container for storing the final list of signal blobs with large enough area and large enough size.
            signal_blobs = np.zeros_like(connected_component_map)
            for label in range(num_components+1):
                # First, select this specific blob, identified by having connected_component_map==label.
                selected_cells = connected_component_map==label
                # if cells with this label are all below threshold, i.e. NONE of them are significant:
                if not any(significant_cells_bool[selected_cells]):
                    pass
                # if cells with this label are all above the threshold, i.e. ALL of them are significant:
                elif all(significant_cells_bool[selected_cells]):
                    if selected_cells.sum()>size_threshold:
                        new_label_used_for_this_chunk = signal_blobs.max()+1
                        signal_blobs[selected_cells] = new_label_used_for_this_chunk
                else:
                    # this condition should never be triggered if programmed correctly.
                    raise ValueError("One of the labelled components include both cells above and below the threshold!!")
            setattr(self, strip_direction+"_blobs", signal_blobs)
        self.size_threshold_used = size_threshold

    def expand_blob(self, new_percentile_threshold=85):
        """
        Tries to expand on the existing large blobs that are detected by label_large_chunks.
        Parameters
        ----------
        new_percentile_threshold: float
            must be less than or equal to percentile_threshold used in self.label_large_chunks.
        radius_of_connectivity: float
            The minimum separation distance below which distinct blobs would be considered as the same, conjoint blob instead.
        """
        for strip_direction in tqdm("uvw", desc="Expanding already-identified blobs:"):
            # Ensure that the self.label_large_chunks had been ran and the self.{u,v,w}_blobs variables had been created.
            if not hasattr(self, strip_direction+"_blobs"):
                raise OperationOrderError(
                    str(self.label_large_chunks)+" must've been ran before to generate the size threshold used."
                    )
            raw_data = getattr(self, strip_direction)
            row_specific_marker = np.percentile(raw_data, new_percentile_threshold, axis=1)
            significant_cells_bool = (raw_data.T>row_specific_marker).T
            setattr(self, strip_direction+"_row_threshold_used", row_specific_marker)

            old_blob_centers = getattr(self, strip_direction+"_blobs")
            connected_component_map, num_components = connected_comp_label(significant_cells_bool, structure=np.ones([3,3]))
            new_signal_blobs = np.zeros_like(connected_component_map)
            for label in range(num_components+1):
                selected_cells = connected_component_map==label
                # choose only the blobs where the master coincides with the
                if any(old_blob_centers[selected_cells]):
                    new_label_used_for_this_chunk = new_signal_blobs.max()+1
                    new_signal_blobs[selected_cells] = new_label_used_for_this_chunk
            setattr(self, strip_direction+"_blobs", new_signal_blobs)

            # now, try to find the outline.

    def trace_outline_simple(self):
        for strip_direction in "uvw":
            if not hasattr(self, strip_direction+"_blobs"):
                raise OperationOrderError("self.{u,v,w}_blobs must've been first populated by self.label_large_chunks and expanded upon using self.expand_blob")
            signal_blobs = getattr(self, strip_direction+"_blobs")
            a = signal_blobs>0
            b = a[::-1] # flipped upside down
            upper_outline = (~np.maximum.accumulate(a, axis=0)).sum(axis=0)
            lower_outline = (np.maximum.accumulate(b, axis=0)).sum(axis=0)
            setattr(self, strip_direction+"_outline", ary([upper_outline, lower_outline]).T)

    def clean_blobs(self, prominence_threshold=1.5):
        """Remove any blobs whose max. prominence does not reach the theshold"""
        for strip_direction in "uvw":
            if not hasattr(self, strip_direction+"_blobs"):
                raise OperationOrderError("self.{u,v,w}_blobs must've been first populated by self.label_large_chunks and expanded upon using self.expand_blob")
            raw_data = getattr(self, strip_direction)
            blob_map = getattr(self, strip_direction+"_blobs")
            row_threshold_used = getattr(self, strip_direction+"_row_threshold_used")
            background = np.broadcast_to(row_threshold_used[:, None], raw_data.shape)

            for blob_label in np.unique(blob_map.flatten()):
                if blob_label==0: # ignore the empty space label.
                    continue
                selected_cells = blob_map==blob_label
                blob_size = selected_cells.sum()
                prominence = (raw_data - background)[selected_cells]
                mean_prominence = prominence.sum()/blob_size
                # max_prominence = prominence.max()
                if mean_prominence<prominence_threshold:
                # if max_prominence<prominence_threshold:
                    blob_map[selected_cells]=0
            setattr(self, strip_direction+"_blobs", blob_map)

    def trace_outline(self, separation_tolerance=2):
        """Trace outline of the blobs"""
        for strip_direction in "uvw":
            if not hasattr(self, strip_direction+"_blobs"):
                raise OperationOrderError("self.{u,v,w}_blobs must've been first populated by self.label_large_chunks and expanded upon using self.expand_blob")
            blob_map = getattr(self, strip_direction+"_blobs")

            for blob_label in np.unique(blob_map.flatten()):
                if blob_label==0:
                    continue
                selected_cells = blob_map==blob_label
                zero_sum_pattern = ary([
                    [ 0, 1, 0],
                    [ 1, 0, 1],
                    [ 0, 1, 0], ])/4
                split_with_neighbours = convolve2d(ary(selected_cells, dtype=int), zero_sum_pattern, )
                is_boundary = split_with_neighbours
                fig, (ax0, ax1) = plt.subplots(2)
                ax1.imshow(is_boundary, alpha=0.5)
                ax0.imshow(getattr(self, strip_direction))
                plt.show()
            # 3-neighbours: concave, should be filled in anyways.
            # 2-neighbours: yes, is edge.
            # 1-neighbour: vertex.
            # draw convex hull with the boundary here?

    def triangulate(self):
        pass

    def u_model(self):
        pass

    def v_model(self):
        pass

    def w_model(self):
        pass

    def get_uncaptured_variations(self):
        pass

    # @classmethod
    # def copy_from(cls):
    #     copy_of_self = cls.__new__()

if __name__=="__main__":
    for num_prongs in range(1,4):
        if num_prongs!=3:
            continue
        for i in range(0,371):
            if i not in (1, 16, 17, 20):
                # Use 3-prong: evt1, 16, 17 .png as the examples.
                continue
            event_name = "train/merged_{}p/evt{}.png".format(num_prongs, i)
            event = Event(event_name)
            fig, (ax0, ax1) = plt.subplots(2)
            fig.suptitle(f"{num_prongs}-prongs evt{i}.png")
            ax0.imshow(ary([event.u, event.v, event.w]).transpose([1,2,0])
                /(len(TRANSLATION_TABLE)-1))
            event.label_large_chunks(97, 40)
            event.expand_blob()
            event.clean_blobs()
            event.trace_outline()
            # event.trace_outline()
            ax1.imshow(ary([ary((event.u_blobs)>0, dtype=float),
                            ary((event.v_blobs)>0, dtype=float),
                            ary((event.w_blobs)>0, dtype=float)]).transpose([1,2,0]))
            ax1.set_title("Extracted Outlines of the Event")
            plt.tight_layout()
            try:
                plt.show()
            except KeyboardInterrupt:
                sys.exit()