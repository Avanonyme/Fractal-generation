import os
import PIL.Image as PILIM
import numpy as np
from Image import IMAGE, COLOUR
from Video import VIDEO
import imageio

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

cmap_dict = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'spring', 'summer', 'autumn', 'winter', 'cool','Wistia',
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu']

raster_image_list=["circle","circle2","fire","human","eyes","planet","stars"]


param={
        #### General parameters
        "clean_dir":False, #clean image dir before rendering
        "verbose":True, #print stuff
        "test":True, #for trying stuff and knowing where t====o put it
        "media_form":"image", #image or video

        "end_dir": "images", #where to put final results
        "temp_dir": "images", #where to put temporary results

        #### Video parameters
        "anim":"explosion pulsing zoom", #pulsing, zoom, translation, flicker, explosion, shading, grain

        # Frame parameters
        "fps":20 ,
        "duration":30, #seconds
        "nb_frames": None, #number of frames, if None, duration and fps are used

        # Animation parameters
        "explosion_param": {"explosion_speed": 45, #log base
                            "start_size": (1,1), #start size in pixels
                            },
        "pulsing_param": {"beta":-0.004, #if None, -25/size
                          "decal": 0,
                          "oscillation_frequency":np.pi/50,
                          "oscillation_amplitude": 10,
                          "c": 3,
                          
                          },
        "translation_param": {"init_damp_r" : 0.4, 
                              "end_damp_r" : 1.25, 
                              "init_damp_c" : -0.5, 
                              "end_damp_c" : 0.75},
        "flicker_param": {"flicker_percentage" : 0.005,
                          "on_fractal" : True, 
                          "dilation_size" : 2,
                          "flicker_amplitude" : 2},
        "grain_param": {"border_thickness": 300,
                        "hole_size": np.ones((3,3)),
                        "distance_exponent_big": 1.2,
                        "distance_exponent_small": 0.6,
                        "nb_rotation":1,
                        },
        "zoom_param": {"zoom_speed":1.02,
                       
                       },
        #### Image parameters
        #General
        "dir": "images",
        "file_name": f"test0",
        "raster_image":np.random.choice(raster_image_list), # if None, raster image is np.zeros((size,size))
        "dpi":2000,

        #Colors
        "color_list":["black","darkgrey","orange","darkred"],
        "cmap":np.random.choice(cmap_dict), #for testing only

        #Shading
        "shading": {"type": "blinn-phong", #None, matplotlib, blinn-phong, fossil
                    "lights": (45., 0, 40., 0, 0.5, 1.2, 1),  # (azimuth, elevation, opacity, k_ambiant, k_diffuse, k_spectral, shininess) for blinn-phong (45., 0, 40., 0, 0.5, 1.2, 1)
                                                                  # (azimuth, elevation, vert_exag, fraction) for matplotlib and fossil (315,20,1.5,1.2)
                    "blend_mode": "hsv",
                    "norm": colors.PowerNorm(0.3),     
                    "nb_rotation": 0.5, #for Dynamic_shading anim
                         },

        #### Fractal parameters
        "method": "RFA Newton",
        "size": 2000,
        "domain":np.array([[-1,1],[-1,1]]),
        ## RFA parameters
        "random":True,

        # Polynomial parameters (Must have value if random==False)
        "degree": 5, #degree of the polynomial
        "func": None,
        "form": "root", #root, coefs, taylor_approx
        "distance_calculation": 4, #see options of get_distance function in RFA_fractals.py
        
        ## Julia parameters

        ## Mandelbrot parameters

        ## Method parameters
        #Newton, Halley
        "itermax":60,
        "tol":1e-8,
        "damping":complex(1.01,-.01),
}



