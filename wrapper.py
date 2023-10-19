import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import sobel, distance_transform_edt
from skimage.feature import canny
import PIL.Image as PILIM
import imageio
import cv2

from Fractal_calculator.Fractal_calculator.Video import VIDEO
from Fractal_calculator.Fractal_calculator.Image import IMAGE, COLOUR
from Fractal_calculator.Fractal_calculator.RFA_fractals import RFA_fractal

def clean_dir(folder, verbose=False):
    if verbose:
        print("Cleaning directory '% s'..." %folder)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                clean_dir(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def paste_image(path_background, path_foreground,img_alpha,img_bg_alpha=None):
    # Read the bckg image
    if isinstance(path_background, str):
        bckg = np.asarray(PILIM.open(path_background).convert('RGBA'))
    else:
        bckg = np.asarray(PILIM.fromarray(path_background.astype(np.uint8)).convert('RGBA')).copy()

    if img_bg_alpha is not None:
        bckg[:,:,3] = img_bg_alpha
    bckg = PILIM.fromarray(bckg.astype(np.uint8))


    # Read the frgrd image
    if isinstance(path_foreground, str):
        frgrd = PILIM.open(path_foreground).convert('RGBA')
    else:
        frgrd = np.asarray(PILIM.fromarray(path_foreground.astype(np.uint8)).convert('RGBA')).copy()

            #Make sure bckg is bigger than frgrd or equal
        if np.asarray(bckg)[:,:,0].size < frgrd[:,:,0].size:
            #if not, put black bcakground around bckg
            bckg = PILIM.fromarray(paste_image(np.zeros_like(frgrd),np.asarray(bckg),np.ones_like(np.asarray(bckg)[:,:,0])*255,img_bg_alpha=np.zeros_like(frgrd[:,:,0])))
        # put alpha values
        frgrd[:,:,3] = img_alpha 

        frgrd = PILIM.fromarray(frgrd)


    # Determine the position to center the frgrd image on the bckg image
    x_offset = (bckg.width - frgrd.width) // 2
    y_offset = (bckg.height - frgrd.height) // 2

    # Paste the frgrd image onto the bckg image using the alpha channel as a mask
    bckg.paste(frgrd, (x_offset, y_offset), frgrd)

    return np.array(bckg)

def create_merged_frames(explosion, vortex, total_frames,verbose=False):
    def duplicate_filepaths(filepaths):
        duplicated_filepaths = []  # Initialize an empty list to store the duplicated filepaths
        for filepath in filepaths:
            # Append the original filepath and its duplicate to the new list
            duplicated_filepaths.extend([filepath, filepath])
        return duplicated_filepaths
    def merge_frames(frames_gif1, frames_gif2, merge_start, total_frames, save_path="merged_frames",verbose=False):
        
        if verbose:
            print("Merging frames...",end="\r")
            print("Merging frames...duplicating",end="\r")
        #frames_gif1 = duplicate_filepaths(frames_gif1)
        print("Merging frames...duplicating done",end="\r")
        frames=[]
        for frame in frames_gif1[:merge_start]:
            #open and append frame to list as array
            frames.append(np.asarray(PILIM.open(frame).convert('RGBA')))

        for i,frame in enumerate(frames_gif1[merge_start:]):

            frames.append(paste_image(frames_gif2[i//4],frame,np.ones_like(np.asarray(PILIM.open(frame).convert('RGBA'))[:,:,0])*255))
        def add_last_frames(total_frames):
            new_frames =[]
            for frame in frames_gif2[i//4:]:
                #we will append the rest of the frames from gif2 in a loop until total_frames
                new_frames.append(np.asarray(PILIM.open(frame).convert('RGBA')))
            return new_frames
        while len(frames) < total_frames:
            print(f"Merging frames...{len(frames)}",end="\r")
            frames+=add_last_frames(total_frames)
        if verbose:
            print("Merging frames...done",end="\r")
        #save frames in new folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i,frame in enumerate(frames[:total_frames]):
            PILIM.fromarray(frame.astype(np.uint8)).save(save_path + f"/frame_{i:0{5}d}.png")
        
        return frames[:total_frames]
            

        
    def make_png_list_from_folder(folder):
        png_list = []
        for file in sorted(os.listdir(folder)):
            if file.endswith(".png"):
                png_list.append(os.path.join(folder, file))
        return png_list

    frames_gif1 = make_png_list_from_folder(os.path.join(os.path.dirname(os.path.dirname(__file__)),f"NFT_cache/sprites/Explosion/Frames/{explosion}"))
    frames_gif2 = make_png_list_from_folder(os.path.join(os.path.dirname(os.path.dirname(__file__)),f"NFT_cache/sprites/Vortex/Frames/{vortex}"))
    merge_start = len(frames_gif1) // 2
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"NFT_cache/sprites/Merged")

    merge_frames(frames_gif1, frames_gif2, merge_start, total_frames, save_path,verbose=verbose)

def empty_cache(cache_dir):
    for sdir in os.listdir(cache_dir):
        for file in os.listdir(os.path.join(cache_dir, sdir)):
            os.remove(os.path.join(cache_dir, sdir, file))
        os.rmdir(os.path.join(cache_dir, sdir))
    for file in os.listdir(cache_dir):
        os.remove(os.path.join(cache_dir, file))

def chunk_to_memmap(arr, chunk_size, directory="memmap_chunks"):
    
    os.makedirs(directory, exist_ok=True)
    
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("The input array should be a 2D square array.")
    
    if chunk_size > arr.shape[0]:
        raise ValueError("Chunk size should be less than or equal to the array size.")

    chunk_files = []
    chunk_indices = []

    # Calculate number of chunks
    num_chunks = arr.shape[0] // chunk_size

    # Split array into chunks
    row_chunks = np.array_split(arr, num_chunks, axis=0)
    dir_list = directory.split(",")
    for i, row_chunk in enumerate(row_chunks):
        col_chunks = np.array_split(row_chunk, num_chunks, axis=1)
        for j, chunk in enumerate(col_chunks):
            filename = os.path.join(directory, f"chunk_{i*chunk_size}_{j*chunk_size}.dat")
            mmapped_chunk = np.memmap(filename, dtype=arr.dtype, mode='w+', shape=chunk.shape)
            np.copyto(mmapped_chunk, chunk)
            mmapped_chunk.flush()
            chunk_files.append(filename)
            chunk_indices.append((i*chunk_size, j*chunk_size))
            
    return chunk_files, chunk_indices

def reassemble_from_memmap(chunk_files, chunk_indices, output_shape, chunk_shape, dtype, directory="memmap_chunks"):
    # Initialize the output array
    reassembled_array = np.empty(output_shape, dtype=dtype)

    if directory is not None:
        # Pre-calculate full paths to avoid repetitive string concatenation
        full_chunk_files = [os.path.join(directory, filename) for filename in chunk_files]
    else:
        full_chunk_files = chunk_files

    # Zip full_chunk_files and chunk_indices together once to avoid multiple iterations
    for filename, (i, j) in zip(full_chunk_files, chunk_indices):
        mmapped_chunk = np.memmap(filename, dtype=dtype, mode='r', shape=chunk_shape)
        # Directly assign the memory-mapped array to the slice of the output array
        reassembled_array[i:i+chunk_shape[0], j:j+chunk_shape[1]] = mmapped_chunk

    return reassembled_array

def save_memmap(array, filename, directory = None, shape = (1,1), dtype = np.float64):
    if not os.path.exists(directory) and directory is not None:
        os.makedirs(directory)

    if directory is not None: # if directory is given, add it to filename
        filename = os.path.join(directory, filename)

    mmapped_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    mmapped_array[:] = array[:]
    mmapped_array.flush()

def process_chunk(frac_obj, compute_method, array_chunk, array_dir, boundary_dir, normal_dir, chunk_size, distance_map, frac_param):
        print(f"Computing chunk {array_chunk}...", end="\r")
        full_chunk_path = os.path.join(array_dir, array_chunk)
        np_chunk = np.memmap(full_chunk_path, dtype=frac_obj.array.dtype, mode='r', shape=(chunk_size, chunk_size))

        # Here we call the pre-determined compute_method
        if compute_method is not None:
            
            z_chunk, conv, dist_chunk, normal_chunk = compute_method(
                np_chunk,
                lambda z: frac_obj.poly.poly(z, frac_obj.coefs),
                lambda z: frac_obj.poly.dpoly(z, frac_obj.coefs),
                lambda z: frac_obj.poly.d2poly(z, frac_obj.coefs),
                frac_param["tol"],
                frac_param["itermax"],
                frac_param["damping"],
                distance_map=distance_map,
                verbose=frac_param["verbose"],
            )

        # Save the chunks
        save_memmap(array=z_chunk, filename=array_chunk, directory=array_dir, shape=np_chunk.shape, dtype=frac_obj.array.dtype)
        save_memmap(array=conv, filename=array_chunk, directory=boundary_dir, shape=np_chunk.shape, dtype=conv.dtype)
        save_memmap(array=normal_chunk, filename=array_chunk, directory=normal_dir, shape=np_chunk.shape, dtype=normal_chunk.dtype)
    
def VIDEO_wrapper_for_fractal(param, im_path_2=None,img_obj = None ,**kwargs):
    #create video object
    video_object = VIDEO(param)
    
    if video_object.verbose:
        print("Video wrapper...",end="")
    
    video_param = param["Video"]
    image_param = param["Image"]
    anim = video_param["animation"]


    print("with anim: ",anim, end="\r")
    ## inputs: 
    if ("zoom" in anim )or ("translate" in anim) or ("shading" in anim):
        frame_list = video_object.Zoom_and_Translate(param, animation = anim, **video_param["zoom_"], **video_param["translate_"])

    else:
        if img_obj is None: # if no image is given, generate one
            img_obj,frame_list = IMAGE_wrapper_for_fractal(param)
            frame_list = (video_object.normalize(frame_list)*255).astype(np.uint8)
            video_object.frac_boundary = [img_obj.frac_boundary] * video_object.nb_frames
        else: # IMAGE object is given, use generated image
            if param["Image"]["return type"] == "iteration":
                frame_list = (video_object.normalize(img_obj.z)*255).astype(np.uint8)    
            elif param["Image"]["return type"] == "distance":
                frame_list = (video_object.normalize(img_obj.shade)*255).astype(np.uint8)
            video_object.frac_boundary = [img_obj.frac_boundary] * video_object.nb_frames
    ## outputs: frame_list

    ## inputs: image or frame_list
    if "pulsing" in anim:
        frame_list = video_object.Pulsing(frame_list,video_object.frac_boundary, **video_param["pulsing_"])        
    if "flicker" in anim and not "distance" in anim:
        frame_list = video_object.Flicker(frame_list,**video_param["flicker_"])
    # add explosion and grain (either this or zoom in image)
    if "explosion" in anim:
        # merge explosion and vortex
        create_merged_frames(param["Image"]["Explosion_image"],param["Image"]["Vortex_image"],video_object.nb_frames,verbose=param["Video"]["verbose"])
        frame_list,alpha_mask = video_object.Grain(frame_list, **video_param["grain_"])
        frame_list = video_object.Explosion(frame_list,alpha_mask, im_path_2, **video_param["explosion_"])
    else: # alpha blending animation
        video_param["flicker_"]["on_fractal"]=False
        frame_list = video_object.Flicker(frame_list,**video_param["flicker_"])
        frame_list = video_object.Alpha(frame_list, im_path_2, render_type=image_param["return type"])
    ## outputs: frame_list with bckg image 
    if video_object.verbose:
        print("Done (V-vm)")
    if param["test"]:
        #save video
        if video_object.frame_save:
            imageio.mimsave(video_object.VID_DIR + "/test.gif", frame_list, fps=video_object.fps)
        else: # list of path
            with imageio.get_writer(video_object.VID_DIR + "/test.gif", mode='I', fps=video_object.fps) as writer:
                for image_path in frame_list:
                    # Read image from disk
                    image = imageio.imread(image_path)
                    
                    # Append the image frame to the GIF
                    writer.append_data(image)

    #add 2 second of only image at the end
    #frame_list = np.append(frame_list, np.array([frame_list[0]]*video_object.fps*5), axis=0)

    return video_object,frame_list

def COLOUR_wrapper(palette_name,method = "accents",**kwargs):
    """
    Wrapper for the COLOUR.create_palette_from_image func. Depending on the method args
    more function can be called

    Args:
        palette_name (str): path to image or name of the palette
        method (str): method(s) name (seaborn, matplotlib, accents,)
        **kwargs: method args
    """
    accent_method = kwargs.pop('accent_method', "complementary")
    simple_cmap = kwargs.pop('simple_palette', False)
    cmap_name  = kwargs.pop('cmap_name', "my_cmap")
    add_black = kwargs.pop('add_black', False)
    

    c_obj = COLOUR()
    if method == "seaborn":
        palette = c_obj.get_seaborn_cmap(palette_name,add_black=add_black)
    elif method == "matplotlib":
        palette = c_obj.get_matplotlib_cmap(palette_name,add_black=add_black)
    elif method == "accents":

        try:
            #open image
            img = np.asarray(PILIM.open(palette_name))

            #create cmap
            palette = c_obj.create_palette_from_image(img)

            palette = c_obj.create_accents_palette(palette,accent_method=accent_method)

            palette = c_obj.create_perceptually_uniform_palette(palette, steps = 256-len(palette) if len(palette)<256 else 2)
            palette = c_obj.create_uniform_colormap(palette)
        except:
            print("accent palette creator has issue. Using random seaborn cmap instead")

            matplotlib_cmap = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'spring', 'summer', 'autumn', 'winter', 'cool','Wistia',
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu']

            seaborn_cmap = ['rocket', 'mako', 'flare', 'crest', 'icefire', 'vlag', 'mako',
                'RdYlGn', 'Spectral']
            
            cmap_choose_from = matplotlib_cmap + seaborn_cmap
            palette = c_obj.get_seaborn_cmap(np.random.choice(cmap_choose_from),add_black=add_black)


    if simple_cmap:
        palette=c_obj.create_palette_from_image(np.asarray(c_obj.render_color_palette(palette, "palettes")))
    else:
        pass

    cmap_name=c_obj.cmap_from_list(palette,cmap_name = cmap_name,add_black=False)

    return cmap_name

def IMAGE_wrapper_for_fractal(param):

    def init_image_object(param):
        image_object = IMAGE(param)

        if image_object.verbose:
            print("Fractal_image...", end="\r")

        frac_param = image_object.set_fractal_parameters(param["Fractal"])

        return image_object, frac_param

    def init_orbit_trap_computing(image_object, frac_param):
        orbit_form, distance_map = None, None  # Initialize to None
        
        if frac_param["raster_image"]:
            try:
                #make sure the orbit path is suitable for the current OS
                orbit_path = os.path.join(image_object.APP_DIR, frac_param["raster_image_dir"], frac_param["raster_image"])
                try:
                    orbit_form = np.array(
                        PILIM.open(orbit_path,)
                        .resize((2000, 2000))
                        .convert("L"),
                        dtype=float,
                    )
                except:
                    print("the error arises from pil")
                orbit_form = np.array(
                    PILIM.open(orbit_path,)
                    .resize((800, 800))
                    .convert("L"),
                    dtype=float,
                )
                distance_map = distance_transform_edt(np.logical_not(orbit_form))
                distance_map = np.divide(
                    distance_map,
                    abs(distance_map),
                    out=np.zeros_like(distance_map),
                    where=distance_map != 0,
                )
            except:
                print("Raster image not found or other issue", os.path.join(image_object.APP_DIR, frac_param["raster_image_dir"], frac_param["raster_image"]))

        return distance_map

    def init_compute_method(frac_param):

        frac_obj = RFA_fractal(frac_param)
        compute_method = None  # Initialize to None


        # RFA
        if "Nova" in frac_param["method"]: # c-mapping before chunking
            c=frac_obj.array #corresponding pixel to complex plane
            c=c.flatten()
            c_coefs=frac_obj.poly.add_c_to_coefs(c,frac_param["func"],frac_param["random"],c_expression=lambda c: np.array([1,(c-1),c-1,1,1,1]))

            print("Computing roots...",end="\r")
            d2roots=np.empty(c_coefs.shape[0],dtype=complex)
            for i in range(c_coefs.shape[0]):
                d2coefs=np.array([],dtype=complex)
                for k in range(2,len(c_coefs[i])):
                    d2coefs=np.append(d2coefs,k*(k-1)*c_coefs[i,k])
                d2roots[i]=np.roots(d2coefs)[0]
            print("Computing roots...Done")
            frac_obj.array=d2roots.reshape(frac_obj.array.shape)
        if "Newton" in frac_param["method"]:
            compute_method = frac_obj.Newton_method
        elif "Halley" in frac_param["method"]:
            compute_method = frac_obj.Halley_method

        # Julia

        # Mandelbrot

        return frac_obj, compute_method

    def init_chunking(image_object, frac_obj):
        if image_object.verbose:
            print("Chunking...", end="\r")

        chunk_size = frac_obj.size // 10
        chunk_dir = os.path.join(image_object.IM_DIR, "memmap")

        array_chunks, array_chunks_indices = chunk_to_memmap(
            frac_obj.array, chunk_size, directory=os.path.join(chunk_dir, "array")
        )
        chunk_to_memmap(
            np.zeros_like(frac_obj.array),
            chunk_size,
            directory=os.path.join(chunk_dir, "boundary"),
        )
        chunk_to_memmap(
            np.zeros_like(frac_obj.array),
            chunk_size,
            directory=os.path.join(chunk_dir, "normal"),
        )

        array_chunks = [os.path.basename(chunk) for chunk in array_chunks]

        if image_object.verbose:
            print("Chunking...Done", end="\r")
        return chunk_size, chunk_dir, array_chunks, array_chunks_indices
    
    def do_shading(image_object, normal):
        if image_object.shading["type"] == "blinn-phong":
            image_object.shade=image_object.blinn_phong(normal,image_object.lights)
        elif image_object.shading["type"] == "matplotlib":
            image_object.shade=image_object.matplotlib_light_source(image_object.z,image_object.lights)
        elif image_object.shading["type"] == "fossil":
            image_object.shade=image_object.matplotlib_light_source(image_object.z*image_object.frac_boundary,image_object.lights)
        else: # we'll return blinn phong by default
            image_object.shade=image_object.blinn_phong(normal,image_object.lights)
        image_object.normal = normal
        return image_object

    def RFA_Image_wrapper(param):

        image_object, frac_param = init_image_object(param)

        distance_map = init_orbit_trap_computing(image_object, frac_param)

        frac_obj, compute_method = init_compute_method(frac_param)
        
        chunk_size, chunk_dir, array_chunks, array_chunks_indices = init_chunking(image_object, frac_obj)

        param["func"] = frac_obj.coefs
        param["form"] = "coefs"
        frac_param["func"] = frac_obj.coefs
        frac_param["form"] = "coefs"
        image_object.func = frac_obj.coefs

        # Precompute directories
        array_dir = os.path.join(chunk_dir, "array")
        boundary_dir = os.path.join(chunk_dir, "boundary")
        normal_dir = os.path.join(chunk_dir, "normal")

        with ProcessPoolExecutor() as executor:
            executor.map(process_chunk, [frac_obj]*len(array_chunks), [compute_method]*len(array_chunks), array_chunks, [array_dir]*len(array_chunks), [boundary_dir]*len(array_chunks), [normal_dir]*len(array_chunks), [chunk_size]*len(array_chunks), [distance_map]*len(array_chunks), [frac_param]*len(array_chunks))

        # Reassemble chunks
        normal = reassemble_from_memmap(chunk_files=array_chunks, chunk_indices=array_chunks_indices, output_shape=frac_obj.array.shape, chunk_shape=(chunk_size,chunk_size), dtype=frac_obj.array.dtype, directory=normal_dir)
        image_object.z = reassemble_from_memmap(chunk_files=array_chunks, chunk_indices=array_chunks_indices, output_shape=frac_obj.array.shape, chunk_shape=(chunk_size,chunk_size), dtype=frac_obj.array.dtype, directory=array_dir)
        conv = reassemble_from_memmap(chunk_files=array_chunks, chunk_indices=array_chunks_indices, output_shape=frac_obj.array.shape, chunk_shape=(chunk_size,chunk_size), dtype=frac_obj.array.dtype, directory=boundary_dir)

        image_object.z = image_object.z.real
        conv = conv.real

        # Edge detection
        image_object.frac_boundary = (canny(conv) + sobel(conv) * (-1) + canny(conv * (-1)) + sobel(conv))
        image_object.frac_boundary = np.where(image_object.frac_boundary > 0, 1, 0)

        # Delete chunks
        empty_cache(chunk_dir)

        #Shading
        do_shading(image_object, normal)

        #Plot
        if param["test"]:
            Image_param = param["Image"]
            # shader
            image_object.Plot(image_object.shade,image_object.file_name+"_shader",Image_param["temp_dir"],print_create=param["verbose"])
            # boundary
            image_object.Plot(image_object.frac_boundary,image_object.file_name+"_nobckg",Image_param["temp_dir"],print_create=param["verbose"])
            # iteration
            image_object.Plot(image_object.z,image_object.file_name+"_iter",Image_param["temp_dir"],print_create=param["verbose"])
            

        if image_object.verbose:
            print("Fractal_image...Done", end="\r")
        
        # Return
        if image_object.return_type  == "iteration":
            return image_object,image_object.z
        elif image_object.return_type  == "distance":
            return image_object,image_object.shade
        elif image_object.return_type  == "boundary":
            return image_object,image_object.frac_boundary

    if "RFA" in param["Fractal"]["method"]:
        return RFA_Image_wrapper(param)
    elif "Julia" in param["Fractal"]["method"]:
        pass
    elif "Mandelbrot" in param["Fractal"]["method"]:
        pass

