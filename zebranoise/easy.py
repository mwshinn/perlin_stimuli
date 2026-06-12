import imageio
from tqdm import tqdm
import warnings
from .util import generate_frames, filter_frames_index_function, apply_filters, discretize
from math import ceil

def zebra_noise(output_file, xsize, ysize, tdur, levels=10, xyscale=.2, tscale=50, fps=30, xscale=1.0, yscale=1.0, seed=0, filters=[("comb", 0.08)]):
    """Generate a .mp4 of zebra noise.

    This method is a simplified interface for the PerlinStimulus class, designed to only generate zebra noise
    as defined in the paper.

    Parameters
    ----------
    output_file : string
        Filename to save the generated .mp4 file
    xsize, ysize : int
        The x and y dimensions of the output video. (Sometimes these will be rounded up to multiples of 16.)
    tdur : float
        The duration of the video in seconds
    levels : int
        The number of octaves to use when approximating the 1/f spectrum. The default of 10 should be more than enough.
    xyscale : float from (0,1)
        The spatial scale of the Perlin noise. Values near 0 will make the video smoother and near 1 choppier.
    tscale : int
        A scaling factor to set the speed of the video
    xscale, yscale : float
        Resize the x and y dimensions of the output
    fps : int
        Frames per second
    seed : int
        Random seed
    
    Returns
    -------
    None, but saves the video file to the desired filename
    """ 
    tsize = int(tdur*fps)
    tscale = tscale * (fps/30)
    textra = (tscale - (tsize % tscale)) % tscale
    if textra > 0:
        warnings.warn(f"Adding {textra} extra timepoints to make tscale a multiple of tdur")
    tsize += round(textra) if (textra % 1 < 1e-5) else ceil(textra)
    get_index = filter_frames_index_function(filters, tsize)
    writer = imageio.get_writer(output_file, fps=fps)
    for _i in tqdm(range(0, tsize)):
        i = get_index(_i)
        frame = generate_frames(xsize, ysize, tsize, [i], levels=levels, xyscale=xyscale, tscale=tscale, xscale=xscale, yscale=yscale, seed=seed)
        filtered = apply_filters(frame, filters, frame_start=_i)
        disc = discretize(filtered[:,:,0])
        writer.append_data(disc)
    writer.close()
