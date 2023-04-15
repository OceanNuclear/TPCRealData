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
zero_sum_quad = ary([
                    [ 0, 1, 0],
                    [ 1, 0, 1],
                    [ 0, 1, 0], ])/4

zero_sum_oct = ary([
                    [ 1, 2, 1],
                    [ 2, 0, 2],
                    [ 1, 2, 1],])/12

class BoundaryCrawler():
    """A crawler that will record the path of the boundary"""
    def __init__(self, blob_map):
        self.blob_map = blob_map
        self.boundary_list = []
        self.interior_node = ary([-1, -1])
        self.exterior_node = ary([-1, -1])

    def initialize_west(self):
        """Initilize a crawler to be at at the east-most position of the map"""
        interior_cells = ary(np.where(self.blob_map)).T
        index_of_start_point_chosen = np.argmin(interior_cells[:, 1])[0]
        interior_cells[index_of_start_point_chosen]
        return

    def step_clockwise(self):
        """Go clockwise around the block"""
        return

def read_raw_pixel_values(filename):
    """Get the raw RBG values"""
    return ary(Image.open(filename, 'r'))

class OperationOrderError(Exception):
    """Operation did not proceed in the required order error"""
    pass

_mnorm = lambda x: x/x.max()

def find_neighbours_index_in_list(remaining_coordinates, cell_of_interest):
    """Given a list of coordinates, and the current cell being studied (cell_of_interest),
    return indices i, j, k ... etc. such that list[i], list[j], list[k]... etc.
    are the coordinates of the cell neighbouring the current cell being studied."""
    L_infty_distance = abs(ary(remaining_coordinates) - cell_of_interest).max(axis=1)
    return np.where(L_infty_distance==1)[0]

def choose_neightbouring_coordinatesL1(coordinate_list, cell_of_interest):
    """Similar to find_neighbours_index_in_list,
    but instead of returning the index (i.e. position) of the closest point in the list,
        it returns the actual coordinate itself.
    This method also uses define neighbour as L1 distance =1 ,i.e. there are 4 neighbours instead of 8."""
    Linfty_distance = abs(ary(coordinate_list) - cell_of_interest).sum(axis=1)
    indices = np.where(Linfty_distance==1)[0]
    return ary(coordinate_list)[indices]

def sort_into_line(list_of_coordinates, selected_area):
    """
    Parameter
    ---------
    A list of integer coordinates, forming an unbroken line.
    """
    # catch the unusual cases (0 points or 1 point) up front.
    if len(list_of_coordinates)<2:
        return list(list_of_coordinates)

    # scroll through the list of boundary points until you find one with exactly 2 neighbours.
    for i in range(len(list_of_coordinates)):
        copied_list = list(list_of_coordinates)
        # Record 1st point.
        output_list = [copied_list.pop(i),]
        neighbour_indices = find_neighbours_index_in_list(copied_list, output_list[-1])
        if len(neighbour_indices)<=2: # between 1-2 neighbours
            chosen_point_index = neighbour_indices[0]
            break
    else:
        raise ValueError("Failed to find a starting point for the outline! Not a contiguous boundary line provided?")

    while len(copied_list)>1:
        # Record next point
        output_list.append(copied_list.pop(chosen_point_index))

        neighbour_indices = find_neighbours_index_in_list(copied_list, output_list[-1])

        if len(neighbour_indices)==1:
            # choose the only point available.
            chosen_point_index = neighbour_indices[0]
            continue
        elif len(neighbour_indices)>1:
            # comparison is required to figure out which one is the closest one.
            empty_space_coords = ary(np.where(~selected_area)[::-1]).T
            empty_neighbours_of_current_point = choose_neightbouring_coordinatesL1(empty_space_coords, output_list[-1])
            for i, nb in zip(neighbour_indices, ary(copied_list)[neighbour_indices]):
                # one and only ONE of these points shall share the thing.
                if len(find_neighbours_index_in_list(empty_neighbours_of_current_point, nb))>0:
                    chosen_point_index = i
                    break
            else:
                raise ValueError("None of the neighbouring boundary point shares a mutual empty cell neighbour with the current point!")
        else: # len(neighbour_indices)<1:
            raise ValueError("No matching neighbours were found")

    return output_list

class Event():
    def __init__(self, filename):
        self.path = filename
        data = read_raw_pixel_values(self.path)
        # Turn the R,G,B channels back into u, v, w.
        u, v, w = data.transpose([2,0,1]) # each channel has [time, strip #]
        self.u, self.v, self.w = vector_table(u), vector_table(v), vector_table(w)

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
                    raise ValueError("One of the labelled components include both cells above and below the threshold!")
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
                # else: # this is already implied
                #     new_signal_blobs[selected_cells] = 0 
            setattr(self, strip_direction+"_blobs", new_signal_blobs)

    def clean_blobs(self, prominence_threshold=1.5, fill_holes=True):
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

            if fill_holes:
                # remove the holes inside of the remaining (definitely trail) blobs using nested for-loops.
                for final_component_label in np.unique(blob_map.flatten()):
                    if final_component_label==0:
                        continue
                    current_component = blob_map==final_component_label
                    background_cells = blob_map==0
                    wetted_by_current_cells = convolve2d(current_component, zero_sum_quad, 'same') # 
                    # as long as the hole is small enough to be completely 'wetted' by the selected cells,
                    # so it must also be fully surrounded by the current component.
                    empty_cell_clusters, num_empty_clusters = connected_comp_label(background_cells)
                    for empty_cluster_label in range(1, num_empty_clusters+1):
                        selected_empty_cells = empty_cell_clusters==empty_cluster_label
                        if all(wetted_by_current_cells[selected_empty_cells]>0):
                            blob_map[selected_empty_cells] = final_component_label
            setattr(self, strip_direction+"_blobs", blob_map)

    def calculate_outline(self, separation_tolerance=2):
        """Trace outline of the blobs"""
        for strip_direction in "uvw":
            if not hasattr(self, strip_direction+"_blobs"):
                raise OperationOrderError("self.{u,v,w}_blobs must've been first populated by self.label_large_chunks and expanded upon using self.expand_blob")
            blob_map = getattr(self, strip_direction+"_blobs")

            outline_lists = []
            for blob_label in np.unique(blob_map.flatten()):
                if blob_label==0:
                    continue
                selected_cells = blob_map==blob_label
                wetted_by_empty_cells = convolve2d(~selected_cells, zero_sum_quad, 'same')>0
                highlighted_boundary = np.logical_and(wetted_by_empty_cells, selected_cells)
                coordinates_of_boundary = ary(np.where(highlighted_boundary)[::-1]).T
                sorted_list_of_boundary = sort_into_line(coordinates_of_boundary, selected_cells)
                outline_lists.append(sorted_list_of_boundary)

                # fig, (ax0, ax1) = plt.subplots(2)
                fig, ax0 = plt.subplots()
                ax0.imshow(getattr(self, strip_direction))
                for line in outline_lists:
                    ax0.plot(*(ary(line).T), color='red')
                # ax1.imshow(is_boundary, alpha=0.5)
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
            event.label_large_chunks(97, 40)
            event.expand_blob()
            event.clean_blobs()
            event.calculate_outline()

            fig, (ax0, ax1) = plt.subplots(2)
            fig.suptitle(f"{num_prongs}-prongs evt{i}.png")
            ax0.imshow(ary([event.u, event.v, event.w]).transpose([1,2,0])
                /(len(TRANSLATION_TABLE)-1))
            ax1.imshow(ary([ary((event.u_blobs)>0, dtype=float),
                            ary((event.v_blobs)>0, dtype=float),
                            ary((event.w_blobs)>0, dtype=float)]).transpose([1,2,0]))
            ax1.set_title("Extracted Outlines of the Event")
            plt.tight_layout()
            try:
                plt.show()
            except KeyboardInterrupt:
                sys.exit()