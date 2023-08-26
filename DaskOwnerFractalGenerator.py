"""
The purpose of this file is to load the parameters from a json file
and to run the fractal generator using the parameters using the dask distributed scheduler

"""

import os
import sys
import json
import dask.distributed as dd
from concurrent.futures import ProcessPoolExecutor


import numpy as np
from scipy.ndimage import sobel, distance_transform_edt, binary_dilation
from skimage.feature import canny
import PIL.Image as PILIM
import imageio

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
                orbit_path = os.path.join(image_object.APP_DIR, "orbit", frac_param["raster_image"] + ".png")
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
                print("Raster image not found or other issue")

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

        array_chunks = [os.path.basename(chunk) for chunk in array_chunks]

        if image_object.verbose:
            print("Chunking...Done")
        return chunk_size, chunk_dir, array_chunks, array_chunks_indices
    
    def do_shading(image_object, normal):
        if image_object.shading["type"] == "blinn-phong":
            image_object.shade=image_object.blinn_phong(normal,image_object.lights)
        elif image_object.shading["type"] == "matplotlib":
            image_object.shade=image_object.matplotlib_light_source(image_object.z,image_object.lights)
        elif image_object.shading["type"] == "fossil":
            image_object.shade=image_object.matplotlib_light_source(image_object.z*image_object.frac_boundary,image_object.lights)
        elif image_object.return_type == "distance": # we'll return blinn phong by default
            image_object.shade=image_object.blinn_phong(normal,image_object.lights)
        image_object.normal = normal
        return image_object

    def RFA_Image_wrapper(param):

        image_object, frac_param = init_image_object(param)

        distance_map = init_orbit_trap_computing(image_object, frac_param)

        frac_obj, compute_method = init_compute_method(frac_param)
        
        chunk_size, chunk_dir, array_chunks, array_chunks_indices = init_chunking(image_object, frac_obj)

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
        image_object=do_shading(image_object, normal)

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

    if "RFA" in param["Fractal"]["method"]:
        return RFA_Image_wrapper(param)
    elif "Julia" in param["Fractal"]["method"]:
        pass
    elif "Mandelbrot" in param["Fractal"]["method"]:
        pass

#test
matplotlib_cmap = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'spring', 'summer', 'autumn', 'winter', 'cool','Wistia',
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu']

seaborn_cmap = ['rocket', 'mako', 'flare', 'crest', 'icefire', 'vlag', 'mako',
                'RdYlGn', 'Spectral']

cmap_dict = matplotlib_cmap + seaborn_cmap

raster_image_list=["circle","circle2","fire","human","eyes","planet","stars"]


param={
        #### General parameters
        "clean_dir":False, #clean image dir before rendering
        "verbose":True, #print stuff
        "test":True, #for trying stuff and knowing where t====o put it

        "end_dir": "images", #where to put final results
        "file_name": f"test0", #name of temp files
        
        "media_form":"image", #image or video

                ## Colour parameters
                        #Colors
        "cmap": "viridis", #test only
        "palette_name":"viridis", #name from cmap_dict, or im_path for custom palette from image
        "color_args":{"method": "matplotlib", #accents, matplotlib, seaborn
                      "simple_palette":False,# if True, nb of colors is scaled down to range(1,10)
                      "accent_method": "split_complementary", #can be combination of complementary, analogous, triadic, split_complementary, tetradicc, shades
                        },

        #### Video parameters
        "Video":{"anim":"explosion", #pulsing, zoom, translation, flicker, explosion, shading, grain

                # Frame parameters
                "frame in memory": False, #if True, frame_list is updated as array, if False, frame_list is updated as list of paths
                "fps":20 ,
                "duration":10, #seconds
                "nb_frames": None, #number of frames, if None, duration and fps are used
                "verbose": True,

                # Animation parameters
                "explosion_": {"explosion_speed": 45, #log base
                                "start_size": (1,1), #start size in pixels
                                },
                "pulsing_": {"beta":-0.004, #if None, -25/size
                                "decal": 0,
                                "oscillation_frequency":np.pi/50,
                                "oscillation_amplitude": 10,
                                "c": 3,
                                
                                },
                "translation_": {"init_damp_r" : 0.4, 
                                "end_damp_r" : 1.25, 
                                "init_damp_c" : -0.5, 
                                "end_damp_c" : 0.75},
                "flicker_": {"flicker_percentage" : 0.005,
                                "on_fractal" : True, 
                                "dilation_size" : 2,
                                "flicker_amplitude" : 2},
                "grain_": {"border_thickness": 300,
                                "hole_size": np.ones((3,3)),
                                "distance_exponent_big": 1.2,
                                "distance_exponent_small": 0.6,
                                "nb_rotation":1,
                                },
                "zoom_": {"zoom_speed":1.02,
                        },
                },
        #### Image parameters
        #General
        "Image":{"dpi":5000,
                 "return type": "iteration", #iteration, distance, boundary

                 "temp_dir": "images", #where to put temporary images, if test is True
                 #Shading
                 "shading": {"type": "blinn-phong", #None, matplotlib, blinn-phong, fossil
                        "lights": (45., 0, 40., 0, 0.5, 1.2, 1),  # (azimuth, elevation, opacity, k_ambiant, k_diffuse, k_spectral, shininess) for blinn-phong (45., 0, 40., 0, 0.5, 1.2, 1)
                                                                        # (azimuth, elevation, vert_exag, fraction) for matplotlib and fossil (315,20,1.5,1.2)
                        "blend_mode": "hsv",
                        "norm": None,     
                        "nb_rotation": 0.5, #for Dynamic_shading anim
                                },
                "verbose": True,
                },
        
        #### Fractal parameters
        "Fractal":{"method": "RFA Newton", #RFA Newton, RFA Halley, 
                "raster_image":"stars", # if None, raster image is np.zeros((size,size))

                "size": 5000,
                "domain":np.array([[-1,1],[-1,1]]),
                "verbose":False,

                ## RFA parameters
                "random":True,

                # Polynomial parameters (Must have value if random==False)
                "degree": 5, #degree of the polynomial
                "func": None,
                "form": "root", #root, coefs, taylor_approx
                "distance_calculation": 4, #see options of get_distance function in RFA_fractals.py

                #Newton, Halley
                "itermax":30,
                "tol":1e-5,
                "damping":complex(1.01,-.01),

                ## Julia parameters

                ## Mandelbrot parameters
        },
        
        
}

if __name__ == "__main__":
    import time
    start = time.time()
    img_test, z = IMAGE_wrapper_for_fractal(param)
    end = time.time()
    print(end - start)
