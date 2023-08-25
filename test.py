import os
import PIL.Image as PILIM
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

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
        "Image":{"dpi":1000,
                 "return type": "iteration", #iteration, distance, boundary

                 "temp_dir": "images", #where to put temporary images, if test is True
                 #Shading
                 "shading": {"type": "blinn-phong", #None, matplotlib, blinn-phong, fossil
                        "lights": (45., 0, 40., 0, 0.5, 1.2, 1),  # (azimuth, elevation, opacity, k_ambiant, k_diffuse, k_spectral, shininess) for blinn-phong (45., 0, 40., 0, 0.5, 1.2, 1)
                                                                        # (azimuth, elevation, vert_exag, fraction) for matplotlib and fossil (315,20,1.5,1.2)
                        "blend_mode": "hsv",
                        "norm": colors.PowerNorm(0.3),     
                        "nb_rotation": 0.5, #for Dynamic_shading anim
                                },
                "verbose": False,
                },
        
        #### Fractal parameters
        "Fractal":{"method": "RFA Newton", #RFA Newton, RFA Halley, 
                "raster_image":np.random.choice(raster_image_list), # if None, raster image is np.zeros((size,size))

                "size": 1000,
                "domain":np.array([[-1,1],[-1,1]]),
                "verbose":True,

                ## RFA parameters
                "random":True,

                # Polynomial parameters (Must have value if random==False)
                "degree": 5, #degree of the polynomial
                "func": None,
                "form": "root", #root, coefs, taylor_approx
                "distance_calculation": 4, #see options of get_distance function in RFA_fractals.py

                #Newton, Halley
                "itermax":60,
                "tol":1e-8,
                "damping":complex(1.01,-.01),

                ## Julia parameters

                ## Mandelbrot parameters
        },
        
        
}

def chunk_to_memmap(arr, chunk_size, directory="memmap_chunks"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("The input array should be a 2D square array.")
    
    if chunk_size > arr.shape[0]:
        raise ValueError("Chunk size should be less than or equal to the array size.")

    chunk_files = []
    chunk_indices = []

    for i in range(0, arr.shape[0], chunk_size):
        for j in range(0, arr.shape[1], chunk_size):
            chunk = arr[i:i+chunk_size, j:j+chunk_size]
            filename = os.path.join(directory, f"chunk_{i}_{j}.dat")
            
            mmapped_chunk = np.memmap(filename, dtype=arr.dtype, mode='w+', shape=chunk.shape)
            mmapped_chunk[:] = chunk[:]
            mmapped_chunk.flush()
            
            chunk_files.append(filename)
            chunk_indices.append((i, j))
            
    return chunk_files, chunk_indices

def reassemble_from_memmap(chunk_files, chunk_indices, output_shape,chunk_shape ,dtype):
    reassembled_array = np.empty(output_shape, dtype=dtype)

    for filename, (i, j) in zip(chunk_files, chunk_indices):
        mmapped_chunk = np.memmap(filename, dtype=dtype, mode='r', shape=chunk_shape)
        shape = mmapped_chunk.shape
        reassembled_array[i:i+shape[0], j:j+shape[1]] = np.array(mmapped_chunk).reshape(shape)

    return reassembled_array
# Create a 4x4 array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]],dtype=np.float64)

# Chunk the array and get the filenames and indices
chunk_files, chunk_indices = chunk_to_memmap(arr, 2)

# Reassemble the array
reassembled_array = reassemble_from_memmap(chunk_files, chunk_indices, arr.shape, (2,2), arr.dtype)

# Validate that the original and reassembled arrays are the same
print(reassembled_array)
print(np.allclose(arr, reassembled_array))  # Should output True
