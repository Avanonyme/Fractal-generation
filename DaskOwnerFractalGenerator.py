"""
The purpose of this file is to load the parameters from a json file
and to run the fractal generator using the parameters using the dask distributed scheduler

"""

import os
import sys
import json
import dask.distributed as dd

import numpy as np
from scipy.ndimage import sobel, distance_transform_edt, binary_dilation
from skimage.feature import canny
import PIL.Image as PILIM

from Video import VIDEO
from Image import IMAGE, COLOUR
from RFA_fractals import RFA_fractal

#used for debugging, and if you want to run the fractal generator on a computer with low memory
def empty_cache(cache_dir):
    for sdir in os.listdir(cache_dir):
        for file in os.listdir(os.path.join(cache_dir, sdir)):
            os.remove(os.path.join(cache_dir, sdir, file))
        os.rmdir(os.path.join(cache_dir, sdir))
    for file in os.listdir(cache_dir):
        os.remove(os.path.join(cache_dir, file))

def load_params(param_path):
    with open(param_path, 'r') as f:
        param = json.load(f)
    return param

def process_json(json):
    """
    call the corresponding functions to process the json file
    """

    #cmap
    json["cmap"] = COLOUR_wrapper(json["palette_name"], json["color_args"])

    if json["media_form"] == "image":
        img_obj,z = IMAGE_wrapper_for_fractal(json)
        
        img_obj.paste_image(json["drawing_path"],z) # TO CHANGE

        img_obj.save_image(json["end_dir"], json["file_name"], json["verbose"])

    elif json["media_form"] == "video":
        video_obj = VIDEO_wrapper_for_fractal(json)
        video_obj.save_video(json["end_dir"], json["file_name"], json["verbose"])

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

def VIDEO_wrapper_for_fractal(param, im_path_2=None,img_obj = None ,**kwargs):
    #create video object
    video_object = VIDEO(param)
    
    if video_object.verbose:
        print("Video maker (V-vm)...",end="")
    
    video_param = param["Video"]
    image_param = param["Image"]
    anim = video_param["anim"]
    print("ANIMATION",anim)

    ## inputs: 
    if ("zoom" in anim )or ("translate" in anim) or ("shading" in anim):
        frame_list = video_object.Zoom_and_Translate(image_param, animation = anim, **video_param["zoom_"], **video_param["translation_"])

    else:
        if img_obj is None: # if no image is given, generate one
            img_obj,frame_list = IMAGE_wrapper_for_fractal(param)
            frame_list = (video_object.normalize(frame_list)*255).astype(np.uint8)
            video_object.frac_boundary = [img_obj.frac_boundary]
        else: # IMAGE object is given, use generated image
            frame_list = (video_object.normalize(img_obj.z)*255).astype(np.uint8)    
            video_object.frac_boundary = [img_obj.frac_boundary]
    ## outputs: frame_list

    ## inputs: image or frame_list
    if "pulsing" in anim:
        frame_list = video_object.Pulsing(frame_list,video_object.frac_boundary, **video_param["pulsing_"])
    if "flicker" in anim:
        frame_list = video_object.Flicker(frame_list,**video_param["flicker_"])
    
    # add explosion and grain (either this or zoom in image)
    if "explosion" in anim:
        frame_list = video_object.Grain(frame_list, **video_param["grain_"])
        frame_list = video_object.Explosion(frame_list, im_path_2, **video_param["explosion_"])
    ## outputs: frame_list

    # zoom in image, replace explosion and grain
    # NOT IMPLEMENTED
   #if "zoom_in" in anim:
   #     frame_list = video_object.Zoom_in(frame_list, **["zoom_in_"])

    if video_object.verbose:
        print("Done (V-vm)")
    #if param["test"]:
    #    #save video
    #    if video_object.frame_save:
    #        imageio.mimsave(video_object.VID_DIR + "/test.gif", frame_list, fps=video_object.fps)
    #    else: # list of path
    #        with imageio.get_writer(video_object.VID_DIR + "/test.gif", mode='I', fps=video_object.fps) as writer:
    #            for image_path in frame_list:
    #                # Read image from disk
    #                image = imageio.imread(image_path)
    #                
    #                # Append the image frame to the GIF
    #                writer.append_data(image)
    return frame_list

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

    c_obj = COLOUR()
    
    if method == "seaborn":
        palette = c_obj.get_seaborn_cmap(palette_name)
    elif method == "matplotlib":
        palette = c_obj.get_matplotlib_cmap(palette_name)
    elif method == "accents":
        #open image
        img = np.asarray(PILIM.open(palette_name))

        #create cmap
        palette = c_obj.create_palette_from_image(img)

        palette = c_obj.create_accents_palette(palette,accent_method=accent_method)

        palette = c_obj.create_perceptually_uniform_palette(palette, steps = 256-len(palette) if len(palette)<256 else 2)
        palette = c_obj.create_uniform_colormap(palette)

    if simple_cmap:
        palette=c_obj.create_palette_from_image(np.asarray(c_obj.render_color_palette(palette, "palettes")))
    else:
        pass

    cmap_name=c_obj.cmap_from_list(palette)

    return cmap_name

def IMAGE_wrapper_for_fractal(param):

    image_object = IMAGE(param)

    if image_object.verbose:
        print("Fractal_image...", end="\r")

    frac_param = image_object.set_fractal_parameters(param["Fractal"])

    orbit_form, distance_map = None, None  # Initialize to None
    
    if frac_param["raster_image"]:
        try:
            orbit_form = np.array(
                PILIM.open(image_object.APP_DIR + "/orbit/" + frac_param["raster_image"] + ".png",)
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
            print("Raster image not found or other issue")

    if "RFA" in frac_param["method"]:
        frac_obj = RFA_fractal(frac_param)
        compute_method = None  # Initialize to None

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


        if image_object.verbose:
            print("Chunking...", end="\r")

        chunk_size = frac_param["N"] // 10
        chunk_dir = os.path.join(image_object.IM_DIR, "memmap")

        array_chunks, array_chunks_indices = chunk_to_memmap(
            frac_obj.array, chunk_size, directory=os.path.join(chunk_dir, "array")
        )

        array_chunks = [os.path.basename(chunk) for chunk in array_chunks]

        if image_object.verbose:
            print("Chunking...Done")

        # Precompute directories
        array_dir = os.path.join(chunk_dir, "array")
        boundary_dir = os.path.join(chunk_dir, "boundary")
        normal_dir = os.path.join(chunk_dir, "normal")

        for i, array_chunk in enumerate(array_chunks):
            if image_object.verbose:
                print(f"Computing chunk {i+1}/{len(array_chunks)}...", end="\r")

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

        # Reassemble chunks
        normal = reassemble_from_memmap(chunk_files=array_chunks, chunk_indices=array_chunks_indices, output_shape=frac_obj.array.shape, chunk_shape=np_chunk.shape, dtype=frac_obj.array.dtype, directory=normal_dir)
        image_object.z = reassemble_from_memmap(chunk_files=array_chunks, chunk_indices=array_chunks_indices, output_shape=frac_obj.array.shape, chunk_shape=np_chunk.shape, dtype=frac_obj.array.dtype, directory=array_dir)
        conv = reassemble_from_memmap(chunk_files=array_chunks, chunk_indices=array_chunks_indices, output_shape=frac_obj.array.shape, chunk_shape=np_chunk.shape, dtype=frac_obj.array.dtype, directory=boundary_dir)

        image_object.z = image_object.z.real
        conv = conv.real

        # Edge detection
        image_object.frac_boundary = (canny(conv) + sobel(conv) * (-1) + canny(conv * (-1)) + sobel(conv))
        image_object.frac_boundary = np.where(image_object.frac_boundary > 0, 1, 0)

        # Delete chunks
        empty_cache(chunk_dir)

        # Julia fractal
    elif "Julia" in frac_param["method"]:
            pass
            
        #Mandelbrot fractal
    elif "Mandelbrot" in frac_param["method"]:
            pass

    #Shading
    if image_object.shading["type"] == "blinn-phong":
        image_object.shade=image_object.blinn_phong(normal,image_object.lights)
    elif image_object.shading["type"] == "matplotlib":
        image_object.shade=image_object.matplotlib_light_source(image_object.z,image_object.lights)
    elif image_object.shading["type"] == "fossil":
        image_object.shade=image_object.matplotlib_light_source(image_object.z*image_object.frac_boundary,image_object.lights)
    elif image_object.return_type == "distance": # we'll return blinn phong by default
        image_object.shade=image_object.blinn_phong(normal,image_object.lights)
    image_object.normal = normal

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
        print("Fractal_image...Done")
    
    # Return
    if image_object.return_type  == "iteration":
        return image_object,image_object.z
    elif image_object.return_type  == "distance":
        return image_object,image_object.shade
    elif image_object.return_type  == "boundary":
        return image_object,image_object.frac_boundary

