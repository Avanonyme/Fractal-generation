import os
import sys
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image as PILIM

import cv2
from scipy import ndimage

from skimage.filters import threshold_mean
from skimage.feature import canny
from skimage.morphology import disk,dilation

import time
from Image import IMAGE

import imageio

def clean_dir(folder, verbose=False):
    import shutil
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


class VIDEO():
    def __init__(self,param) -> None:
        print("Init Videos class(V-Init) fractals...")

        ### Create folders
        self.APP_DIR = os.path.dirname(os.path.abspath(__file__))
        self.IM_DIR=os.path.join(self.APP_DIR,param["temp_dir"])
        try: os.mkdir(self.IM_DIR)
        except: pass
        self.VID_DIR=os.path.join(self.IM_DIR,"video")
        try: os.mkdir(self.VID_DIR)
        except: pass

        self.FRAME_DIR=os.path.join(self.VID_DIR,"frames") + "/"
        try: os.mkdir(self.FRAME_DIR)
        except: pass
        clean_dir(self.FRAME_DIR)

        ### Set paramaters
        self.set_video_parameters(param)

        print("Done (V-Init)")

    def set_video_parameters(self,param):
        self.fps=param["fps"]
        self.duration=param["duration"]
        self.nb_frames = param["nb_frames"]

        temp_im = IMAGE(param)
        self.cmap = temp_im.cmap

        self.frac_boundary = []

        self.verbose = param["verbose"]


    ## ARRAY HANDLING
    def init_array(self,N,domain):
        """create array of complex numbers"""
        real_dom=np.linspace(domain[0,0],domain[0,1],N,endpoint=False) #expanded domain
        complex_dom=np.linspace(domain[1,0],domain[1,1],N,endpoint=False)
        return np.array([(item+complex_dom*1j) for i,item in enumerate(real_dom)]).reshape(N,N).transpose() #array of shape (N,N)

    def circle_mask(self,z,dpi,r):
        a,b=dpi//2,dpi//2
        r=r
        n=z.shape[0]

        y,x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r

        return mask
    
    def normalize(self,img_obj):
        """
        Normalize image or list of images between 0 and 1
        if img_obj is a list, normalize each image in the list

        handle of true_divide error
        """
        # ignore divide by 0 error
        np.seterr(divide='ignore', invalid='ignore')

        if isinstance(img_obj,list):
            result = [(arr - np.min(arr)) / (np.max(arr) - np.min(arr)) for arr in img_obj]
            #check for true_divide error
            for i,arr in enumerate(result):
                if np.isnan(arr).any():
                    result[i] = np.zeros(arr.shape)

        else:
            result = (img_obj - np.min(img_obj)) / (np.max(img_obj) - np.min(img_obj))
            #check for true_divide error
            if np.isnan(result).any(): #any because divide by 0 error says max = min
                result = np.zeros(result.shape)
        
        # reset divide by 0 error
        np.seterr(divide='warn', invalid='warn')
        return result
        
    def convert_list(self,img_list, dtype):
        
        new_img_list = []
        for img in img_list:
            img = img.astype(dtype)
            new_img_list.append(img)
        return new_img_list
        
    def paste_image(self, path_background, path_foreground,img_alpha):
        # Read the bckg image
        if isinstance(path_background, str):
            bckg = PILIM.open(path_background).convert('RGBA')
        else:
            bckg = PILIM.fromarray(path_background.astype(np.uint8)).convert('RGBA')

        # Read the frgrd image
        if isinstance(path_foreground, str):
            frgrd = PILIM.open(path_foreground).convert('RGBA')
        else:
            frgrd = np.asarray(PILIM.fromarray(path_foreground.astype(np.uint8)).convert('RGBA')).copy()
            # put alpha values
            frgrd[:,:,3] = img_alpha

            frgrd = PILIM.fromarray(frgrd)
        

        # Determine the position to center the frgrd image on the bckg image
        x_offset = (bckg.width - frgrd.width) // 2
        y_offset = (bckg.height - frgrd.height) // 2

        # Paste the frgrd image onto the bckg image using the alpha channel as a mask
        bckg.paste(frgrd, (x_offset, y_offset), frgrd)

        return np.array(bckg)
    ### VIDEO MAKER ###
    def Video_maker(self,param, im_path_2=None, **kwargs):
        if self.verbose:
            print("Video maker (V-vm)...",end="")

        anim = param["anim"]

        ## inputs: param
        if "zoom" or "translate" or "shading" in anim:
            frame_list = self.Zoom_and_Translate(param, animation = anim, **param["zoom_param"], **param["translation_param"])

        else:
            img_obj = IMAGE(param)
            frame_list = img_obj.Fractal_image()
            self.frac_boundary = img_obj.frac_boundary
        ## outputs: frame_list

        ## inputs: image or frame_list
        if "pulsing" in anim:
            frame_list = self.Pulsing(frame_list,self.frac_boundary, **param["pulsing_param"])
        if "flicker" in anim:
            frame_list = self.Flicker(frame_list,**param["flicker_param"])
        
        # add explosion and grain (either this or zoom in image)
        if "explosion" in anim:
            frame_list = self.Grain(frame_list, **param["grain_param"])
            frame_list = self.Explosion(frame_list, im_path_2, **param["explosion_param"])
        ## outputs: frame_list

        # zoom in image
        if "zoom_in" in anim:
            frame_list = self.Zoom_in(frame_list, **param["zoom_in_param"])

        ## make video
        
        if self.verbose:
            print("Done (V-vm)")


    ### ANIMATIONS ###
    def Explosion(self,img_obj,im_path_2=None, **kwargs):
        '''Generate explosion animation of input image
        img_obj: image or images to be animated
        im_path_2: put explosion on top of this image, if None, explosion is on top of black background
        '''
        # get the parameters
        log_base = kwargs.get('explosion_speed', 45)
        inf_size = kwargs.get('start_size', (1,1))
        resample = kwargs.get('resample_method', 3)
        nb_frames = len(img_obj)

        # get the size of the image
        if isinstance(img_obj,list):
            sup_size=(img_obj[0].shape[0],img_obj[0].shape[1])
        else:
            sup_size=(img_obj.shape[0],img_obj.shape[1])
        
        # get the list of sizes of the explosion
        list_ex=list(np.unique(np.clip((np.logspace(np.log(inf_size[0]),np.log(sup_size[0]),num = 50, base = log_base)).astype(int),1,sup_size[0])))
        explosion_size = len(list_ex)

        temp_list = list(np.ones((nb_frames - 2*len(list_ex) -20)) * sup_size[0]) # 20 because we want animation to start before end of explosion
        temp_list+=list(np.flip(list_ex))
        list_ex +=temp_list
        # get the background image
        if im_path_2 is None:
            #black background
            im_bg = np.zeros((sup_size[1],sup_size[0],3))
        else:
            #image background
            im_bg = np.asarray(PILIM.open(im_path_2).resize(sup_size))
        
        # loop over the images, resize, and add it on top of the background
        frame_list = []
        # while explosion, frame is similar to previous frame and resizing
        for i,size in enumerate(list_ex):
            if self.verbose:
                print("explosion anim: ",i,"/",len(list_ex), end="\r")
            
            if isinstance(img_obj,list) and size == sup_size[0] or not (i<explosion_size-10 or i>explosion_size+10): #middle frames
                print(" middle frames",end="")
                img = img_obj[i]
            elif isinstance(img_obj,list) and size != sup_size[0] and (i<explosion_size-10 or i>explosion_size+10): #start and end frames
                #check if size is smaller than previous size
                if size < list_ex[i-1]: #size is smaller than previous size, we're in the shrinking phase (end)
                    print(" we're in the shrinking phase (end)",end="")
                    img = img_obj[i-1]
                else: #size is bigger than previous size, we're in the explosion phase (beginning)
                    print(" we're in the explosion phase (beginning)",end="")
                    img = img_obj[i+1]
            else:
                print(" else",end="")
                img = img_obj


            img_alpha = np.where(img == 0, 0,1)*255
            img_alpha = np.asarray(PILIM.fromarray(img_alpha.astype(np.uint8)).resize((size,size),resample=resample)) if size != sup_size[0] else img_alpha
            
            plt.imsave(self.FRAME_DIR + f"frame_{i}.png",img, cmap = self.cmap, vmin=0,vmax=255) #image must be in RGB or RGBA format
            im = np.asarray(PILIM.open(self.FRAME_DIR + f"frame_{i}.png").resize((size,size),resample=resample)) if size != sup_size[0] else np.asarray(PILIM.open(self.FRAME_DIR + f"frame_{i}.png"))

            new_im = self.paste_image(im_bg,im,img_alpha)

            frame_list.append(new_im) #list of arrays (n,n,3)
            
        # while not explosion, we just save the frame
        return frame_list
    
    def Grain(self,img, **kwargs):
        """
        Apply granular gfilter to an image

        img: a numpy array of shape (height, width, 3)
        border_thickness: the thickness of the granular gfilter from the border (pixels)
        hole_size: the size of the holes (radius of the disk structuring element)

        return: a numpy array of shape (height, width, 3)
        """
        if self.verbose:
            print("grain anim...",end="")
        # get the parameters
        border_thickness = kwargs.get('border_thickness', 200)
        hole_size = kwargs.get('hole_size', np.ones((1,1)))
        n_frames = kwargs.get('nb_frames', self.fps * self.duration)
        distance_exponent_big = kwargs.get('distance_exponent', 0.3)
        distance_exponent_small = kwargs.get('distance_exponent', 0.3)

        nb_rotation = kwargs.get('nb_rotation', 1)

        def do_grain(mask_border,fill_value=0.3,distance_exponent=0.3):
            """
            grainification of an image

            """
            # Apply distance transform
            dist_transform = cv2.distanceTransform(mask_border.astype(np.uint8), cv2.DIST_L2, 5)
            dist_transform = np.float_power(dist_transform * (1 + dist_transform / np.max(dist_transform)), distance_exponent)

            # Normalize the distance transform image for viewing
            mask = (cv2.normalize(dist_transform, None, alpha=fill_value, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
            # create the granular gfilter
            gfilter = np.zeros(mask.shape)
            gfilter = ndimage.binary_dilation(np.random.rand(*mask.shape) <= mask, structure= hole_size)
            return gfilter

        def grain_fill(gfilter,fill_value=0.3):
            """
            fill empty space in  gfilter with more grain
            """
            dilation=ndimage.binary_fill_holes(gfilter.copy())

            # Apply distance transform
            dist_transform = np.float_power(cv2.distanceTransform(dilation.astype(np.uint8), cv2.DIST_LABEL_PIXEL, 5),0.5)
            # Normalize the distance transform image for viewing
            mask = (cv2.normalize(dist_transform, None, alpha=fill_value, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
            # create the granular gfilter
            new_filter = np.zeros(mask.shape)
            new_filter = np.random.rand(*mask.shape) <= mask

            return new_filter
        if isinstance(img,list):
            n_frames = len(img)
            width,height=img[0].shape[0],img[0].shape[1]
        else:
            width,height=img.shape[0],img.shape[1]

        # Hide center from gfilter
        mask=np.ones((width,height))
        mask[border_thickness:-border_thickness,border_thickness:-border_thickness]=0
        mask_border=np.copy(mask)

        #smaller mask
        small_gfilter=np.ones((width,height))
        border_thickness_small=border_thickness//2
        small_gfilter[border_thickness_small:-border_thickness_small,border_thickness_small:-border_thickness_small]=0

        gfilter=do_grain(mask_border,fill_value=0.3,distance_exponent=distance_exponent_big)
        small_gfilter=do_grain(small_gfilter,fill_value= 0.1,distance_exponent=distance_exponent_small)

        small_gfilter=small_gfilter.astype(bool)
        # Apply the granular gfilter on the distance transform
        gfilter = np.logical_not(gfilter).astype(np.float64)
        
        frame_list=[]
        
        #Animate rotation of grain
        new_gfilter=np.zeros((width,height))

        if isinstance(img,list): #frame list
            n_frames = len(img)
            angles = np.around(np.linspace(0, 360 * nb_rotation, n_frames),2)

            for n,image in enumerate(img):
                i = angles[n]
                print("rotation grain anim: ",int(i),"/", 360 * nb_rotation, end="\r")
                gfilter_rotated=ndimage.rotate(gfilter,i,reshape=False,cval=0,prefilter=False,mode="constant")
                new_gfilter+=gfilter_rotated-0.3*new_gfilter

                new_gfilter+=grain_fill(new_gfilter,fill_value=0.0).astype(np.float64)
                #new_gfilter %= 10

                new_gfilter = np.clip(new_gfilter,0,1.2)

                new_img=image * new_gfilter
            
                frame_list.append((self.normalize(new_img) * 255).astype(np.uint8))


        else:
            angles = np.around(np.linspace(0, 360 * nb_rotation, n_frames),2)
            for i in angles:
                
                if self.verbose:
                    print("rotation grain anim: ",int(i),"/", 360 * nb_rotation, end="\r")
                gfilter_rotated=ndimage.rotate(gfilter,i,reshape=False,cval=0,prefilter=False,mode="constant")
                new_gfilter+=gfilter_rotated-0.3*new_gfilter

                new_gfilter+=grain_fill(new_gfilter,fill_value=0.0).astype(np.float64)
                #new_gfilter %= 10

                new_gfilter = np.clip(new_gfilter,0,1.2)

                new_img=img * new_gfilter
            
                frame_list.append((self.normalize(new_img) * 255).astype(np.uint8))
        if self.verbose:
            print('grain anim done')
        return frame_list #int 0-255 list of array (n,n)
    
    def Flicker(self,img_obj,**kwargs):
        """
        Apply a flicker animation to an image

        img: a numpy array of shape (height, width, 3), or a list of such arrays
        
        flicker_amplitude: the amplitude of the flicker effect 
        flicker_percentage: the percentage of pixels to flicker
        dilation_size: the size of the dilation kernel
        nb_frames: the number of frames in the animation (single image only)
        on_fractal: whether to apply the flicker on the fractal or everywhere

        return: a list of numpy arrays of shape [frames,(height, width, 3)]
        """
        # Get the parameters
        dilation_size = kwargs.get("dilation_size", 2)
        flicker_amplitude = kwargs.get("flicker_amplitude", 0.9)
        flicker_percentage = kwargs.get("flicker_percentage", 0.0005)
        nb_frames = kwargs.get("nb_frames", self.fps * self.duration)
        on_fractal = kwargs.get("on_fractal", False)

        if self.verbose:
            print("Flicker animation...", end="")
        # We'll store each frame as we create it
        frame_list = []
        # Calculate the total number of pixels
        if isinstance(img_obj,list):
            total_pixels = img_obj[0].shape[0] * img_obj[0].shape[1]
            width,height=img_obj[0].shape[0],img_obj[0].shape[1]
        else: 
            total_pixels = img_obj.shape[0] * img_obj.shape[1]
            width,height=img_obj.shape[0],img_obj.shape[1]
        num_flicker_pixels = int(total_pixels * flicker_percentage)

        # Generate the indices of the pixels to flicker
        flicker_indices = np.random.choice(total_pixels, num_flicker_pixels, replace=False)

        #Create mask from flicker indices
        mask=np.zeros((width,height))
        np.put(mask,flicker_indices,1)
        mask=(mask).astype(bool)    

        mask=ndimage.binary_dilation(mask,iterations=dilation_size)

        #update flicker indices
        flicker_indices=np.where(mask==1)

        # give random [-2pi,2pi] value to phase
        phase = np.random.uniform(-2*np.pi,2*np.pi,mask.shape)

        
        
        # Apply flicker effect on fractal only, if specified
        if on_fractal:
            
            if isinstance(img_obj,list):
                #copy mask n times, where n is the number of frames
                mask = [mask.copy() for i in range(nb_frames)]
                for i,img_array in enumerate(img_obj):
                    fractal_mask=self.frac_boundary[i]
                    fractal_mask = np.where(fractal_mask>0.5,1,0).astype(bool)

                    mask[i]*=fractal_mask

            else:
                fractal_mask=self.frac_boundary
                fractal_mask = np.where(fractal_mask>0.5,1,0).astype(bool)

                mask*=fractal_mask

        pulse_func = lambda x,phase: np.sin(x * 2*np.pi + phase ) + 1 #sinusoidal flicker,
        pulse_func = np.vectorize(pulse_func)

        if isinstance(img_obj,list): #multiple images
            
            for i,img_array in enumerate(img_obj):
                
                img_array = img_array.copy()

                # Calculate the sine value for each pixel (including phase)
                sine_values = np.sin((i / nb_frames * 2 * np.pi) + phase)

                # Apply the mask to the sine values
                sine_values_masked = sine_values * mask[i]

                # Calculate the flicker multiplier
                flicker_multiplier = 1 + (flicker_amplitude * sine_values_masked)

                # Apply on image
                img_array = img_array * flicker_multiplier

                # Normalize the image
                img_array = (self.normalize(img_array) * 255).astype(np.uint8)

                # Add the image to the frame list
                frame_list.append(img_array)

               
            
        else: #single image

            # Specify the number of frames you want in the animation
            num_frames = nb_frames
            for i in range(num_frames+1):
                
                img_array = img_obj.copy()

                # Calculate the sine value for each pixel (including phase)
                sine_values = np.sin((i / nb_frames * 2 * np.pi) + phase)

                # Apply the mask to the sine values
                sine_values_masked = sine_values * mask

                # Calculate the flicker multiplier
                flicker_multiplier = 1 + (flicker_amplitude * sine_values_masked)

                # Apply on image
                img_array *= flicker_multiplier

                # Normalize the image
                img_array = (self.normalize(img_array) * 255).astype(np.uint8)

                # Add the image to the frame list
                frame_list.append(img_array)

        if self.verbose:
            print("flicker anim done")

        return frame_list #int 0 - 255 list of arrays(n,n)

    def Pulsing(self,img_obj,fractal_bounds,**kwargs):
        """ 
        Create a pulsing animation of the fractal image

        img_obj: single array or list of arrays
        frac_boundary: array of the fractal boundary, if None, it is img_obj.frac_boundary (img_obj must be an IMAGE object)

        kwargs
        beta: damping coefficient for oscillations
        decal: number of pixels to add to the image to avoid the animation to be cut when saving as GIF
        cmap: colormap to use for the animation
        """
        # get the parameters
        beta = kwargs.get("beta",-0.3)
        decal = kwargs.get("decal",0)
        cmap = kwargs.get("cmap",self.cmap)
        omega = kwargs.get("oscillation_frequency",np.pi)
        amplitude = kwargs.get("amplitude",1)
        c = kwargs.get("c",None)

        if self.verbose:
            print("Pulsing...")

        def f(u,beta = -0.03):
            return amplitude * np.exp(- beta * u) * np.sin(omega * u)
        
        if isinstance(img_obj,list):
            img = img_obj[0]

            frac_size = img.shape[0]
            # Time
            max_t = len(img_obj)

            #speed of the wave
            if c is None:
                c = (frac_size + decal)/max_t
        else:
            img = img_obj

            frac_size = img_obj.shape[0]
            
            if c is None:
                c = frac_size / 300  # speed of the wave
            # Time
            max_t = self.fps*self.duration if self.nb_frames is None else self.nb_frames
        
        if beta is None:
            beta = - 25 / frac_size
        
        # Gif properties
        # Source location
        x_center, y_center = frac_size // 2, frac_size // 2
        # Wave properties



        # frame array
        frame_list = []

        X, Y = np.meshgrid(np.arange(frac_size), np.arange(frac_size))
        R = np.sqrt((X - x_center)**2 + (Y - y_center)**2) #distance

        Mask = np.zeros((frac_size, frac_size))
        
        for step,t in enumerate(np.arange(0, max_t, 1)):

            Mask = np.where(R<= c*t,1,0 ) # where c*t is peak of a Gaussian wave
            Psi = f(R - c * t,beta=beta)

            if isinstance(img_obj,list):
                try: #fractal bounds is a list
                    wave_im = (((Psi * Mask) * fractal_bounds[step] * 255)).astype(np.uint8)
                except: #fractal bounds is a list containing one array
                    wave_im = (((Psi * Mask) * fractal_bounds[0] * 255)).astype(np.uint8)

                img = (img_obj[step] * 255/np.max(img_obj[step]))
                
                #new_im = (self.normalize(wave_im + img_obj[step]) * 255).astype(np.uint8)

            else: #single image
                wave_im = (((Psi * Mask) * fractal_bounds[0] * 255)).astype(np.uint8)
                
                img = (img * 255/np.max(img)) # image always appears
                
                #new_im = (self.normalize(wave_im + img) * 255).astype(np.uint8)
                
                
            frame_list.append((wave_im + img).astype(np.uint8))
            if self.verbose:
                print("  ",step,"/",max_t,np.max(Psi) ,end="\r")
        if self.verbose:
            print("Pulsing done")
        return frame_list #int 0 - 255 list of arrays(n,n)

    def Zoom_and_Translate(self,param, animation = "zoom translate shading", **kwargs):
        """
        Create a zoom and/or complex translation animation of the fractal image
        
        param: dict of the parameters of the fractal
        zoom: boolean, if True, zoom in the fractal
        translate: boolean, if True, translate the fractal

        kwargs
        init_damp_r: initial damping coefficient for oscillations
        end_damp_r: final damping coefficient for oscillations
        init_damp_c: initial complex damping coefficient for oscillations
        end_damp_c: final complex damping coefficient for oscillations
        
        """
        if self.verbose:
            print("(Vm-Zoom_and_Translate)...")
            if 'zoom' in animation:
                print("Zooming...",end=" ")
            if "translate" in animation:
                print("Translating...")
            if "shading" in animation:
                print("Shading...",end=" ")
        # get the parameters
        init_damp_r = kwargs.get("init_damp_r",0.4)
        end_damp_r = kwargs.get("end_damp_r",1.35)
        init_damp_c = kwargs.get("init_damp_c",-0.5)
        end_damp_c = kwargs.get("end_damp_c",0.85)

        zoom_speed_factor = kwargs.get("zoom_speed",1.1)


        nb_frames = self.fps*self.duration if self.nb_frames is None else self.nb_frames
        if self.nb_frames is not None:
            self.fps = 20
            self.duration = self.nb_frames//self.fps


        # we'll save the frames in a list
        frame_list = []

        # get damping list from the parameters
        if "translate" in animation:
            damping_list=np.linspace(init_damp_r,end_damp_r,self.fps*self.duration+1)+np.linspace(init_damp_c,end_damp_c,self.fps*self.duration+1)*1j
            param["damping"]=damping_list[0]

        if "shading" in animation:
            nb_rotation = kwargs.get("nb_rotation",1)
            azimuth = np.linspace(0,22.5 * nb_rotation, nb_frames)

        def check_coord(z,edges,dpi,coord,zoom,prev_point):
            #init old coord
            array=self.init_array(dpi,coord)

            for i in range(1,int(z.size),1):
                mask=self.circle_mask(z,dpi,i)
                points=np.where(edges==mask,mask,np.zeros_like(mask))
                if np.any(points)==True:
                    candidate=np.where(points==True)
                    point=[candidate[0][0],candidate[1][0]]
                    break
            try:
                pts=array[point[0],point[1]]
            except:
                return coord*zoom,prev_point
            pts=[pts.real,pts.imag]

            coord=np.array([[(pts[0])-1*zoom,(pts[0])+1*zoom], #real
                        [(pts[1])-1*zoom,(pts[1])+1*zoom]]) #complex
            return coord,point
        
        #loop over frames
        zoom_speed = 1
        for _ in range(nb_frames):
        # num_frames is self.fps*self.duration

            print("Zoom and Translate and Shading",_,end="\r")
            #Create frame
            Imobj=IMAGE(param) 
            im = Imobj.Fractal_image()
            # update parameters
            #_==0
            if _ == 0:
                print("Zoom and Translate and Shading",_,end="")
                param["form"]=None
                param["random"]=False #True only for first frame at most
                param["pts"] = [0,0]

                if self.verbose:
                    #turn it off for specific function cause it's annoying
                    self.verbose = False
                    remind_me_to_turn_it_back_on = True
                param["verbose"] = False

            param["func"]= Imobj.func
            
            if "zoom" in animation:
                zoom_speed = zoom_speed/zoom_speed_factor
                param["domain"],param["pts"]=check_coord(im,Imobj.frac_boundary,param["dpi"],param["domain"],zoom_speed,prev_point=param["pts"])
            if "translate" in animation:
                print('kespass')
                param["damping"]=damping_list[_]

            
            #save frames
            if "zoom" or "translate" in animation:
                self.frac_boundary.append(Imobj.frac_boundary)

            if "shading" in animation:
                shade_im = self.Dynamic_shading(Imobj,azimuth = [azimuth[_]] ,**kwargs)[0]
                frame_list.append((self.normalize(shade_im) * 255).astype(np.uint8))
            else:
                frame_list.append((self.normalize(im) * 255).astype(np.uint8))

        if self.verbose:
            print("Done (Vm Zoom_and_Translate)")
        if remind_me_to_turn_it_back_on:
            self.verbose = True

        return frame_list #int 0 - 255 list of arrays(n,n)
    
    def Dynamic_shading(self, img_obj ,**kwargs):
        """
        Create a day passing effect using normal of the fractal image and a light source

        img_obj: IMAGE object

        kwargs:
        light: tuple of the light source coordinates
        shader_type: type of shading (blinn-phong, matplotlib)

        nb_rotation: number of complete rotation of the light source
        
        
        """

        if self.verbose:
            print("(Vm-Dynamic_shading)...")
        # get the parameters
        light = kwargs.get("light",(45., 0, 40., 0, 0.5, 1.2, 1))
        shader_type = kwargs.get("type","blinn-phong")
        nb_rotation = kwargs.get("nb_rotation",1)
        blend_mode = kwargs.get("blend_mode","normal")
        norm = kwargs.get("norm",colors.PowerNorm(0.3))
        

        nb_frames = self.fps*self.duration if self.nb_frames is None else self.nb_frames
        if self.nb_frames is not None:
            self.fps = 20
            self.duration = self.nb_frames//self.fps


        
        # we'll save the frames in a list
        frame_list = []

        #loop over frames
        azimuth_list = kwargs.pop("azimuth",np.linspace(0,22.5 * nb_rotation, nb_frames))
        for azimuth in azimuth_list:
            light = list(light)
            light[0] = azimuth
            light = tuple(light)
            print(" ",azimuth,end="\r")

                #Shading
            if shader_type == "blinn-phong":
                shade=img_obj.blinn_phong(img_obj.normal,light)
            elif shader_type == "matplotlib":
                shade=img_obj.matplotlib_light_source(img_obj.z,light,blend_mode=blend_mode,norm=norm)
            elif shader_type == "fossil":
                shade=img_obj.matplotlib_light_source(img_obj.z*img_obj.frac_boundary,light,blend_mode=blend_mode,norm=norm)
            else:
                print("Shader type not recognized or None (Vm Dynamic_shading)")
                return img_obj.z

            #save frame
            frame_list.append((self.normalize(shade) * 255).astype(np.uint8))

        if self.verbose:
            print("Done (Vm Dynamic_shading)")
        return frame_list
        
        
if __name__=='__main__':
    i=0
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
    "file_name": f"test{i}",
    "raster_image":np.random.choice(raster_image_list), # if None, raster image is np.zeros((size,size))
    "dpi":500,
    #Colors
    "color_list":[],

    #Shading
    "shading":True,
    "lights": (45., 0, 40., 0, 0.5, 1.2, 1),
    #Filters

    #### Fractal parameters
    "method": "RFA Newton",
    "size": 500,
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

    ##Julia parameters
}

    try:
        obj=VIDEO(param)
        obj.Video_maker(param)

    except KeyboardInterrupt:
        sys.exit()
        