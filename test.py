import os
import PIL.Image as PILIM
import numpy as np
from Image import IMAGE


cmap_dict = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'spring', 'summer', 'autumn', 'winter', 'cool','Wistia',
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu']

raster_image_list=["circle","circle2","fire","human","eyes","planet","stars"]

param={
    #### General parameters
    "clean_dir":False, #clean image dir before rendering
    "verbose":False, #print stuff
    "test":True, #for trying stuff and knowing where to put it
    "media_form":"video", #image or video

    #### Animation parameters
    "anim method":"explosion", #pulsing, zoom, translation, rotation
    "frame_number":1,
    "frame_size": 2160,
    "fps":20 ,
    "duration":3, #seconds
    "zoom":1.2, #if animation method is zoom 

    #### Image parameters
    #General
    "cmap":np.random.choice(cmap_dict), #for testing only   
    "dir": "images",
    "file_name": f"test0",
    "raster_image":np.random.choice(raster_image_list), # if None, raster image is np.zeros((size,size))
    "dpi":1000,
    #Colors
    "color_list":[],

    #Shading
    "shading":True,
    "lights": (45., 0, 40., 0, 0.5, 1.2, 1),
    #Filters

    #### Fractal parameters
    "method": "RFA Newton",
    "size": 200,
    "domain":np.array([[-1.,1.],[-1.,1.]]),
    ## RFA paramters
    "random":True,
    # Polynomial parameters (Must have value if random==False)
    "degree": None, #random.randint(5,20),
    "func": None,#[1,-1/2+np.sqrt(3)/2*1j,-1/2-np.sqrt(3)/2*1j,1/2+np.sqrt(3)/2*1j,1/2-np.sqrt(3)/2*1j], 
    "form": "", #root, coefs, taylor_approx

    "distance_calculation": 4, #see options of get_distance function in RFA_fractals.py
    
    #Method parameters
    "itermax":50,
    "tol":1e-8,
    "damping":complex(0.2,-.01),
}

