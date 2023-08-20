import os
import PIL.Image as PILIM
import numpy as np
from Image import IMAGE
from Video import VIDEO
import imageio

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
        "test":True, #for trying stuff and knowing where to put it
        "media_form":"image", #image or video

        "end_dir": "images", #where to put final results
        "temp_dir": "images", #where to put temporary results

        #### Video parameters
        "anim method":"explosion zoom", #pulsing, zoom, translation, flicker, explosion

        # Frame parameters
        "fps":20 ,
        "duration":3, #seconds
        "nb_frames": None, #number of frames, if None, duration and fps are used

        "pulsing_param": {"beta": 0.5,
                          "decal": 0},
        "translation_param": {"init_damp_r" : 0.4, 
                              "end_damp_r" : 1.35, 
                              "init_damp_c" : -0.5, 
                              "end_damp_c" : 0.85},
        "flicker_param": {"flicker_percentage" : 0.0005,
                          "on_fractal" : False, 
                          "dilation_size" : 2,
                          "flicker_amplitude" : 0.9},
        "explosion_param": {"border_thickness": 200,
                            "hole_size": 3},
        "zoom_param": {"zoom_speed":1.02},
        #### Image parameters
        #General
        "dir": "images",
        "file_name": f"test0",
        "raster_image":np.random.choice(raster_image_list), # if None, raster image is np.zeros((size,size))
        "dpi":500,

        #Colors
        "color_list":["black","darkgrey","orange","darkred"],
        "cmap":np.random.choice(cmap_dict), #for testing only

        #Shading
        "shading": {"type": "blinn-phong", #None, matplotlib, blinn-phong, fossil
                    "lights": (45., 0, 40., 0, 0.5, 1.2, 1),  # (azimuth, elevation, opacity, k_ambiant, k_diffuse, k_spectral, shininess) for blinn-phong
                                                                  # (azimuth, elevation, vert_exag, fraction) for matplotlib
                    "blend_mode": "hsv",
                    "norm": colors.PowerNorm(0.3),     
                         },

        #### Fractal parameters
        "method": "RFA Newton",
        "size": 500,
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
        "itermax":50,
        "tol":1e-6,
        "damping":complex(1.01,-.01),
}
