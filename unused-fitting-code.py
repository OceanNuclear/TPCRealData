"""
Done:
Steps for processing each image:
[x]Remove horizontal streaks (by performing fourier analysis in the vertical direction)
[x]Set background noise level
[x]Extract ALL curvilinear parts, by various techniques, e.g.:
    [x]use the peak prominence algorithm on each channel, but adapted into 2D (time x strip #)
    [x]but reframed to something more complicated like:
        - determine the connected component for each level (ignoring spurious holes and dots smaller than a certain size)
        - Pick the ridge of each peak as the trail.
[x]Get the true outline of each blob
    - THINK OF IT AS A SPANNING TREE!
    - fill out all of the holes first.
    - less-than-fully-connected nodes should be near the edge! remove them?
    - Use the 'wildfire' algorithm!
    [x]OH that was easy, just convolve and then grep for all of the values !=0 or 1.
[ ]Get the rough backbones of each blob, by burning away the edges.
        - Or I can use argmax within that circle. Might be a bit ugly but it'll work.
        [ ]resolution = 0.5 pixel?
[x]Model it as some sort of distribution? Nah.
- Goal: extract the ridges, despite the noise.
____
# exciting skimage functions to try:
transform.
    iradon
    iradon_sart
    ifrt2
    hough_line
    hough_line_peaks
    probabilistic_hough_line
[x]segmentation.morphological_chan_vese
filters.
    [o]frangi # detect continuous ridges, e.g. vessels, wrinkles, rivers.
    [o]butterworth
    [x]meijering # detect continuous ridges, e.g. neurites, wrinkles, rivers.
    [x]sato # detect continuous ridges, e.g. tubes, wrinkles, rivers.
    # sato does clean it up a bit.
    [x]hessian # detect continuous edges, e.g. vessels, wrinkles, rivers.

    [x]apply_hysteresis_threshold # I can make the mask extraction better using hystersis
    [x]difference_of_gaussians # and I want to try this

graph.
    # find the actual paths using one of these
    MCP
    MCP_*
    shortest_path
    route_through_array
    central_pixel
morphology.
    thin
    skeletonize
    medial_axis
# I can use line extractor after the skeletonize:
morphology.
    [x](isotropic_/binary_/)erosion
    [x](isotropic_/binary_/)opening
    [x](isotropic_/binary_/)closing
    [x](isotropic_/binary_/)dilation
    [x]reconstruction
    # See https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html#sphx-glr-auto-examples-applications-plot-morphology-py
    [x]h_minima
    [x]h_maxima
    [x]max_tree
    remove_small_holes
    remove_small_objects

draw.
    [o]polygon2mask
    [x]polygon_perimeter
restoration.
    [x]ellipsoid_kernel
segmentation.
    [x]flood
    [x]flood_fill
____
[x]Objective 1: find the lengthscale:
    - hard coded
____
[x]Approach 2: Burn away the edges
    [ ]This can also give us useful information about the width of the track, i.e. along the ridge how many steps of burning is required before the fire reaches the centre.
    - This is actually known as the Euclidean distance transform.
    - I don't need this anymore because there's skeletonize.

"""

"""Tested ideas:
Potential workflow 1a.:
# filters.
    # optionally, splice in old_analysis_procedure()
    frangi(sigmas=range(4,18,2), black_ridges=False, mode='reflect'),
    meijerling(sigmas=range(4,18,2), black_ridges=False, mode='reflect'),
    butterworth(cutoff_frequency_ratio=0.05,
                high_pass=False,
                order=4,
                squared_butterworth=True),

    filters.apply_hysteresis_threshold()
All has worse performance than the existing workflow 1.
"""

class EventDeprecated():
    def denoise_and_highlight(self, method, argument_generator):
        """
        Parameters
        ----------
        method: function (usually skimage.*.* function)
            Takes in the raw u/v/w values, and output enhanced values which accentuate the trail while deminishing the trail.
        argument_generator: function: dict
            Generates a dictionary of the arguments to be supplied to ``method''

        Returns
        -------
        sets the u_enhanced, v_enhanced, w_enhanced attribute to self.
        """
        for strip_direction in tqdm("uvw", desc=f"Applying {method.__name__} on each strip direction"):
            raw_data = getattr(self, strip_direction)
            argument_dict = argument_generator(raw_data)
            enhanced_data = method(raw_data, **argument_dict)
            setattr(self, strip_direction+"_enhanced", enhanced_data)

    def new_threshold_method(self, thresholder):
        """
        Parameter
        ---------
        thresholder: function(np.array[int])-> np.array[bool]
        """
        for strip_direction in "uvw":
            enhanced_data = getattr(self, strip_direction+"_enhanced")
            significant_cells = thresholder(enhanced_data)
            blob_map, num_components = connected_comp_label(significant_cells, structure=np.ones([3,3]))
            setattr(self, strip_direction+"_blobs", connected_component_map)


if __name__=="__main__":
    def event_name_getter(num_prongs, num_evt):
        event_name = "train/merged_{}p/evt{}.png".format(num_prongs, num_evt)
        return event_name

    def load_event(num_prongs, num_evt):
        event_name = event_name_getter(num_prongs, num_evt)
        return Event(event_name)

    for num_prongs in range(1,4):
        if num_prongs!=3:
            continue
        for i in range(0,371):
            # if i not in (1, 16, 17, 20,):
                # Use 3-prong: evt1, 16, 17 .png as the examples.
            if i>20:
                continue
            if i<8:
                continue
            def analysis_procedure2(evt):
                
                method = skimage.filters.frangi
                evt.denoise_and_highlight(method,
                    lambda _x: dict(sigmas=range(4,18,2), black_ridges=False, mode='nearest')
                    )
                
                method = skimage.filters.meijering
                evt.denoise_and_highlight(method,
                    lambda _x: dict(sigmas=range(4,16,2), black_ridges=False, mode='reflect')
                    )
                
                method = skimage.filters.butterworth
                evt.denoise_and_highlight(method,
                    lambda _x: dict(
                        cutoff_frequency_ratio=0.05,
                        high_pass=False,
                        order=4,
                        squared_butterworth=True)
                    )
                ax1.imshow( ary([event.u_enhanced, event.v_enhanced, event.w_enhanced]
                                ).transpose([1,2,0])
                            /(len(TRANSLATION_TABLE)-1)
                            )
                ax1.set_title( f"After applying filter {method.__name__}" )
                ax1.set_xlabel("time (bin)")
                ax1.set_ylabel("strip number")
                # evt.new_threshold_method()
                # evt.calculate_outline()
            fig, (ax0, ax1) = plt.subplots(1, 2)
            analysis_procedure2(event)