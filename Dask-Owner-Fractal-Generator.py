"""
The purpose of this file is to load the parameters from a json file
and to run the fractal generator using the parameters using the dask distributed scheduler

"""

import os
import sys
import json
import dask.distributed as dd

import numpy as np
from skimage.feature import canny, sobel
import PIL.Image as PILIM

from Video import VIDEO
from Image import IMAGE, COLOUR
from RFA_fractals import RFA_fractal


#used for debugging, and if you want to run the fractal generator on a computer with low memory
def empty_cache(cache_dir):
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
        print("Fractal_image...",end="\r")

    ### Set param
    frac_param=image_object.set_fractal_parameters(param["Fractal"])

    if frac_param["raster_image"]!="" or frac_param["raster_image"] is not None:
        try:
            orbit_form=np.array(PILIM.open(image_object.APP_DIR+"/orbit/"+frac_param["raster_image"]+".png",).resize((frac_param["N"]+1,frac_param["N"]+1)).convert("L"),dtype=float)
        
        except:
            print("Raster image",frac_param["raster_image"],"not found. \nIf you do not want to use a raster image, set 'raster_image' parameters to ''.\n Else check the name of the image in the 'orbit' folder")

    ### General type of Fractal
    if "RFA" in frac_param["method"]: #Root finding algorithm
        frac_obj=RFA_fractal(frac_param)
        frac_param["func"]=frac_obj.coefs
        image_object.func=frac_obj.coefs

        ## Subtype of RFA Fractal - Method will be called in THIS loop
        #NOT WORKING RN
        if "Nova" in frac_param["method"]: # NO VARIATION OF METHOD FOR SUBTYPES YET
            """
            M-set fractal like with c-mapping

            Results are...experimental, and i cant be bothered to fix it
            Use with caution
            """
            c=frac_obj.array #corresponding pixel to complex plane
            shape=c.shape;c=c.flatten()

            c_coefs=frac_obj.poly.add_c_to_coefs(c,frac_param["func"],frac_param["random"],c_expression=lambda c: np.array([1,(c-1),c-1,1,1,1]))

            #assuming that the c-length is on axis 0
            print("Computing roots...",end="\r")
            d2roots=np.empty(c_coefs.shape[0],dtype=complex)
            for i in range(c_coefs.shape[0]):
                d2coefs=np.array([],dtype=complex)
                for k in range(2,len(c_coefs[i])):
                    d2coefs=np.append(d2coefs,k*(k-1)*c_coefs[i,k])
                d2roots[i]=np.roots(d2coefs)[0]
            print("Computing roots...Done")

        #print(d2roots.shape,c.shape) #should be equal

            if "Newton" in frac_param["method"]:

                image_object.z,conv,dist,normal=frac_obj.Newton_method(d2roots, #z0
                                                        lambda z: np.array([frac_obj.poly.poly(z[i],c_coefs[i]) for i in range(len(c_coefs))]), #f
                                                        lambda z: np.array([frac_obj.poly.dpoly(z[i],c_coefs[i]) for i in range(len(c_coefs))]), #f'
                                                        tol=1.e-05,
                                                        max_steps=50,
                                                        damping=complex(1,0.2),
                                                        verbose = frac_param["verbose"],)


                image_object.z,conv,dist=image_object.z.reshape(shape),conv.reshape(shape),dist.reshape(shape)

                image_object.z,conv,dist=image_object.z.real,conv.real,dist.real
  
        ## No subtype specified
        else:
            if "Newton" in frac_param["method"]: #Newton method
                image_object.z,conv,dist,normal=frac_obj.Newton_method(frac_obj.array,
                                                    lambda z: frac_obj.poly.poly(z,frac_obj.coefs),
                                                    lambda z: frac_obj.poly.dpoly(z,frac_obj.coefs),
                                                    lambda z: frac_obj.poly.d2poly(z,frac_obj.coefs),
                                                    frac_param["tol"],
                                                    frac_param["itermax"],
                                                    frac_param["damping"],
                                                    orbit_form=orbit_form,
                                                    verbose = frac_param["verbose"],)

            
            elif "Halley" in frac_param["method"]:
                image_object.z,conv,dist,normal=frac_obj.Halley_method(frac_obj.array,
                                                    lambda z: frac_obj.poly.poly(z,frac_obj.coefs),
                                                    lambda z: frac_obj.poly.dpoly(z,frac_obj.coefs),
                                                    lambda z: frac_obj.poly.d2poly(z,frac_obj.coefs),
                                                    frac_param["tol"],
                                                    frac_param["itermax"],
                                                    frac_param["damping"],
                                                    orbit_form=orbit_form,
                                                    verbose = frac_param["verbose"],)
        #throw away imaginary part
        image_object.z,conv=image_object.z.real,conv.real

        # Julia fractal
    elif "Julia" in frac_param["method"]:
            pass
            
        #Mandelbrot fractal
    elif "Mandelbrot" in frac_param["method"]:
            pass

    #Edge detection
    image_object.frac_boundary=(canny(conv)+sobel(conv)*(-1) + canny(conv*(-1))+sobel(conv)) # +1e-02 to avoid division by 0
    image_object.frac_boundary = np.where(image_object.frac_boundary>0,1,0)

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
    #if param["test"]:
    #    Image_param = param["Image"]
    #    # shader
    #    image_object.Plot(image_object.shade,image_object.file_name+"_shader",Image_param["temp_dir"],print_create=param["verbose"])
    #    # boundary
    #    image_object.Plot(image_object.frac_boundary,image_object.file_name+"_nobckg",Image_param["temp_dir"],print_create=param["verbose"])
    #    # iteration
    #    image_object.Plot(image_object.z,image_object.file_name+"_iter",Image_param["temp_dir"],print_create=param["verbose"])
        

    if image_object.verbose:
        print("Fractal_image...Done")

    if image_object.return_type  == "iteration":
        return image_object,image_object.z
    elif image_object.return_type  == "distance":
        return image_object,image_object.shade
    elif image_object.return_type  == "boundary":
        return image_object,image_object.frac_boundary
