import os
import sys
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
from PIL import Image as PILIM

import cv2
from scipy import ndimage

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


class VIDEO():
    def __init__(self,param) -> None:
        print("Init Videos class(V-Init) fractals...",end="\r")

        from daskowner import IMAGE_wrapper_for_fractal
        self.IMAGE_wrapper_for_fractal = IMAGE_wrapper_for_fractal
        ### Create folders
        self.APP_DIR = os.path.dirname(os.path.abspath(__file__))
        self.IM_DIR=os.path.join(self.APP_DIR,param["Image"]["temp_dir"])
        try: os.mkdir(self.IM_DIR)
        except: pass
        self.VID_DIR=os.path.join(self.APP_DIR,os.path.dirname(param["Image"]["temp_dir"]),"video")
        try: os.mkdir(self.VID_DIR)
        except: pass

        self.FRAME_DIR=os.path.join(self.VID_DIR,"frames") + "/"
        try: os.mkdir(self.FRAME_DIR)
        except: pass
        clean_dir(self.FRAME_DIR)

        ### Set paramaters
        self.set_video_parameters(param["Video"])
        self.dpi=param["Image"]["dpi"]
        self.cmap = param["cmap"]

        print("Init Videos class(V-Init) fractals...Done\t\t",end="\r")

    def set_video_parameters(self,param):
        self.fps=param["fps"]
        self.duration=param["duration"]
        self.nb_frames = param["nb_frames"] if param["nb_frames"] is not None else self.fps * self.duration

        self.frac_boundary = []

        self.verbose = param["verbose"]

        self.frame_save = param["frame in memory"]

        self.fractal_background = param["fractal background"]

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
                if np.isinf(arr).any():
                    result[i] = np.ones(arr.shape)

        else:
            result = (img_obj - np.min(img_obj)) / (np.max(img_obj) - np.min(img_obj))
            #check for true_divide error
            if np.isnan(result).any(): #any because divide by 0 error says max = min
                result = np.zeros(result.shape)
            if np.isinf(result).any():
                result = np.ones(result.shape)
        
        # reset divide by 0 error
        np.seterr(divide='warn', invalid='warn')
        return result
        
    def convert_list(self,img_list, dtype):
        
        new_img_list = []
        for img in img_list:
            img = img.astype(dtype)
            new_img_list.append(img)
        return new_img_list
    
    def create_disk_mask(self,diameter, shape):
        # Determine the center of the disk
        center_y, center_x = shape[0] // 2, shape[1] // 2

        # Create an array of indices representing the grid
        y, x = np.ogrid[:shape[0], :shape[1]]

        # Calculate the squared distance from each point to the center
        distance_squared = (x - center_x)**2 + (y - center_y)**2

        # Create the mask where points with squared distance less than or equal to (diameter/2)^2 are True
        mask = distance_squared <= (diameter/2)**2

        return mask
    
        
    def apply_colormap_to_grayscale_image(self,image_path,type_= "path"):
        """
        Open a grayscale image, apply a Matplotlib colormap, and save as an array in memory.

        Parameters:
            image_path (str): The path to the grayscale image to be opened.
            cmap_name (str): The name of the Matplotlib colormap to be applied.

        Returns:
            colored_image (ndarray): An array representing the image with the colormap applied.
        """

        # Step 1: Read the grayscale image
        if type_ == "path":
            grayscale_image = mpimg.imread(image_path)
        else:
            grayscale_image = image_path.copy()

        # Step 2: Make sure the image is grayscale
        if len(grayscale_image.shape) == 3:
            raise ValueError("The input image is not grayscale.")

        # Step 3: Apply the colormap
        cmap = plt.get_cmap(self.cmap)
        colored_image = cmap(grayscale_image / np.max(grayscale_image))

        # Step 4: Convert to uint8 [0, 255] scale (optional)
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

        return colored_image
    
    def paste_image(self, path_background, path_foreground,img_alpha,img_bg_alpha=None):
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
                bckg = PILIM.fromarray(self.paste_image(np.zeros_like(frgrd),np.asarray(bckg),np.ones_like(np.asarray(bckg)[:,:,0])*255,img_bg_alpha=np.zeros_like(frgrd[:,:,0])))
            # put alpha values
            frgrd[:,:,3] = img_alpha 

            frgrd = PILIM.fromarray(frgrd)


        # Determine the position to center the frgrd image on the bckg image
        x_offset = (bckg.width - frgrd.width) // 2
        y_offset = (bckg.height - frgrd.height) // 2

        # Paste the frgrd image onto the bckg image using the alpha channel as a mask
        bckg.paste(frgrd, (x_offset, y_offset), frgrd)

        return np.array(bckg)
    ### ANIMATIONS ###
    def Alpha(self,img_obj,im_path_2=None, render_type = "iteration",**kwargs):
        """
        Gradual reveal of the image on top of an image
        
        """

        if self.verbose:
            print("Alpha anim...",end="\r")

        # get the size of the image
        if isinstance(img_obj,list):
            if isinstance(img_obj[0],str): #img_obj is list of path
                width,height = np.asarray(PILIM.open(img_obj[0]).convert("L")).shape[0],np.asarray(PILIM.open(img_obj[0]).convert("L")).shape[1]

            else:#list of arrays
                width,height=img_obj[0].shape[0],img_obj[0].shape[1]
                
        else: # single image
            if isinstance(img_obj,str): #img_obj is path
                width,height = np.asarray(PILIM.open(img_obj).convert("L")).shape[0],np.asarray(PILIM.open(img_obj).convert("L")).shape[1]

            else: #img_obj is array
                width,height=img_obj.shape[0],img_obj.shape[1]

        if im_path_2 is None:
            #black background
            im_bg = np.zeros((img_obj.shape[0],img_obj.shape[1],3)) if not isinstance(img_obj,list) else np.zeros((width,height,3))
        
        else:
            #image background
            im_bg = np.asarray(PILIM.open(im_path_2).resize((width,height))) if not isinstance(img_obj,list) else np.asarray(PILIM.open(im_path_2).resize((width,height)))
        # get the parameters

        # set alpha channel
        img_alpha= np.zeros((width,height))

        # loop over the images
        frame_list = []

        for i in range(self.nb_frames):
            if self.verbose:
                print("Alpha anim...",i,"/",self.nb_frames, end="\r")

            if isinstance(img_obj,list):    

                img = self.apply_colormap_to_grayscale_image(img_obj[i]) if isinstance(img_obj[0],str) else self.apply_colormap_to_grayscale_image(img_obj[i])

                boundary = self.frac_boundary[i] if not isinstance(img_obj[0],str) else self.frac_boundary
            else:
                img = self.apply_colormap_to_grayscale_image(img_obj, type_ = "array")   
                boundary = self.frac_boundary
            #for the first seconds, only alpha on fractal augments if render type is iteration
            if render_type == "distance":
                # alpha increase is homogeneous
                if i < (self.nb_frames//12):
                    img_alpha = img_alpha + 1/(self.nb_frames//12) *255
                #for the last seconds, decrease alpha rapidly
                elif i > self.nb_frames - self.nb_frames//36:
                    img_alpha = img_alpha - 1/(self.nb_frames//18) *255
                else: #pass 
                    img_alpha = np.ones((width,height)) * 255
            else: #render type is iteration
                if i < self.nb_frames//12:
                    img_alpha = np.where(boundary == 0, 0, 255 * i/(self.nb_frames//12))
                #for the last seconds, decrease alpha rapidly
                elif i > self.nb_frames - self.nb_frames//24:
                    img_alpha = np.where(boundary == 0, 255-(255*i/self.nb_frames//24), 255 - 255 * (i - (self.nb_frames - self.nb_frames//24))/(self.nb_frames//24))
                else:
                    img_alpha = np.ones((width,height)) * 255
            
            img_alpha[boundary == 1] = np.clip(img_alpha[boundary == 1],60,210)
            img_alpha[boundary==0] = np.clip(img_alpha[boundary==0],100,230)
                
            
            if self.fractal_background:
                #save and open fractal image
                plt.imsave(self.FRAME_DIR + f"frame_{i}.png",img.astype(np.uint8), cmap = self.cmap, vmin=0,vmax=255) #image must be in RGB or RGBA format
                im = np.asarray(PILIM.open(self.FRAME_DIR + f"frame_{i}.png").resize((width,height)))[:,:,:3]
                

                new_im = self.paste_image(im,im_bg,im_bg[:,:,-1], img_bg_alpha = img_alpha) #im_bg should have alpha channel

            else:
                #save and open fractal image
                plt.imsave(self.FRAME_DIR + f"frame_{i}.png",img.astype(np.uint8), cmap = self.cmap, vmin=0,vmax=255) #image must be in RGB or RGBA format
                im = np.asarray(PILIM.open(self.FRAME_DIR + f"frame_{i}.png").resize((width,height)))

                #paste image
                new_im = self.paste_image(im_bg,im,img_alpha)
            if self.frame_save:
                frame_list.append(new_im.astype(np.uint8)) # list of arrays (n,n,4)
            else:
                plt.imsave(self.FRAME_DIR + f"frame_{i}.png",new_im.astype(np.uint8), cmap = self.cmap, vmin=0,vmax=255) #image must be in RGBA format

                frame_list.append(self.FRAME_DIR + f"frame_{i}.png") #list of paths
        
        if self.verbose:
            print("Alpha anim...",self.nb_frames,"/",self.nb_frames,"Done\t\t")

        return frame_list

    def Explosion(self,img_obj,alpha_mask,im_path_2=None, **kwargs):
        '''Generate explosion animation of input image
        img_obj: image or images to be animated
        im_path_2: put explosion on top of this image, if None, explosion is on top of black background
        '''

        if self.verbose:
            print("Explosion anim...",end="\r")
        # get the parameters
        log_base = kwargs.get('explosion_speed', 45)
        inf_size = kwargs.get('start_size', (1,1))
        resample = kwargs.get('resample_method', 3)
        nb_frames = len(img_obj)
        # get the size of the image
        if isinstance(img_obj,list):
            if isinstance(img_obj[0],str): #self.frame_save is true and img_obj is list of path
                sup_size = np.asarray(PILIM.open(img_obj[0]).convert("L")).shape
            else: #self.frame_save is false and img_obj is list of arrays
                sup_size=(img_obj[0].shape[0],img_obj[0].shape[1])
        else: #img_obj is array
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
        
        #sprite handling
        def make_png_list_from_folder(folder):
            png_list = []
            for file in sorted(os.listdir(folder)):
                if file.endswith(".png"):
                    png_list.append(os.path.join(folder, file))
            return png_list
        sprite_list = make_png_list_from_folder(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"NFT_cache/sprites/Merged"))
        # match sprite list lenght to list_ex length
        #if len(sprite_list) > len(list_ex):
        #    sprite_list = sprite_list[:len(list_ex)]
        #elif len(sprite_list) < len(list_ex):
        #    sprite_list += sprite_list[-(len(list_ex)-len(sprite_list)):]
        
        # loop over the images, resize, and add it on top of the background
        frame_list = []
        # while explosion, frame is similar to previous frame and resizing
        for i,size in enumerate(list_ex):
            size = int(size)
            if self.verbose:
                print("Explosion anim...",i,"/",len(list_ex), end="\r")
            

            if isinstance(img_obj,list) and size == sup_size[0] or not (i<explosion_size-10 or i>explosion_size+10): #middle frames
                img = img_obj[i]
            elif isinstance(img_obj,list) and size != sup_size[0] and (i<explosion_size-10 or i>explosion_size+10): #start and end frames
                #check if size is smaller than previous size
                if size < list_ex[i-1]: #size is smaller than previous size, we're in the shrinking phase (end)
                    img = img_obj[i-1] if not isinstance(img_obj[0],str) else img_obj[i]
                else: #size is bigger than previous size, we're in the explosion phase (beginning)
                    img = img_obj[i+1] if not isinstance(img_obj[0],str) else img_obj[i]
            else: #single image
                img = img_obj

            if self.fractal_background:
                im = PILIM.open(img).resize((size,size),resample=resample) if size != sup_size[0] else PILIM.open(img)
                im = self.apply_colormap_to_grayscale_image(np.asarray(im), type_ = "array")

                im_bg_sprite = PILIM.open(sprite_list[i]).resize((size,size),resample=resample) if i > explosion_size else PILIM.open(sprite_list[i]).resize(sup_size,resample=resample)
                im_bg_sprite = np.asarray(im_bg_sprite)
                im_bg_sprite = self.past_image(im_bg_sprite,im,im_bg[:,:,-1], img_bg_alpha = np.ones_like(im_bg[:,:,-1])*255) #im_bg should have alpha channel
                new_im = self.paste_image(im_bg_sprite,im_bg,im_bg[:,:,-1], img_bg_alpha = np.ones_like(im_bg[:,:,-1])*255) #im_bg should have alpha channel
            
            else:
                if not isinstance(img_obj[0],str): #img_obj is list of arrays 
                    img_alpha = np.asarray(PILIM.fromarray(alpha_mask[i].astype(np.uint8)).resize((size,size),resample=resample)) if size != sup_size[0] else alpha_mask[i]
                    #normalize to 255
                    img_alpha = np.clip(self.normalize(img_alpha) * 255,0,230)
                
                    plt.imsave(self.FRAME_DIR + f"frame_{i}.png",img, cmap = self.cmap, vmin=0,vmax=255, dpi =self.dpi) #image must be in RGB or RGBA format
                    im = np.asarray(PILIM.open(self.FRAME_DIR + f"frame_{i}.png").resize((size,size),resample=resample)) if size != sup_size[0] else np.asarray(PILIM.open(self.FRAME_DIR + f"frame_{i}.png"))
                    im = self.apply_colormap_to_grayscale_image(im, type_ = "array")
                else: #img_obj is list of paths
                    im = PILIM.open(img).resize((size,size),resample=resample) if size != sup_size[0] else PILIM.open(img)
                    img_alpha = np.asarray(PILIM.fromarray(alpha_mask[i].astype(np.uint8)).resize((size,size),resample=resample)) if size != sup_size[0] else alpha_mask[i]
                    #normalize to 255
                    img_alpha = np.clip(self.normalize(img_alpha) * 255,0,230)
                    im = self.apply_colormap_to_grayscale_image(np.asarray(im), type_ = "array")
                    
                im_bg_sprite = PILIM.open(sprite_list[i]).resize((size,size),resample=resample) if i > explosion_size else PILIM.open(sprite_list[i]).resize((sup_size[0],sup_size[1]),resample=resample)
                im_bg_sprite = np.asarray(im_bg_sprite)
                im_bg_sprite = self.paste_image(im_bg,im_bg_sprite,im_bg_sprite[:,:,-1])

                new_im = self.paste_image(im_bg_sprite,im,img_alpha)            
            if self.frame_save:
                frame_list.append(new_im) # list of arrays (n,n,4)
            else:
                plt.imsave(self.FRAME_DIR + f"frame_{i}.png",new_im, cmap = self.cmap, vmin=0,vmax=255) #image must be in RGBA format
                frame_list.append(self.FRAME_DIR + f"frame_{i}.png") #list of path

        # add im_bg frame to end of list
        if self.frame_save:
            frame_list + ([im_bg]*self.fps*4) #save im_bg as array for 2 seconds
        else:
            frame_list + ([im_path_2]*self.fps*4) #save im_bg as path for 2 seconds
        # while not explosion, we just save the frame
        if self.verbose:
            print("Explosion anim...",len(list_ex),"/",len(list_ex),"Done\t\t")
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
            print("Grain anim...",end="\r")
        # get the parameters
        border_thickness = kwargs.get('border_thickness', 200)
        hole_size = np.ones(kwargs.get('hole_size', (1,1)))
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
            if isinstance(img[0],str): #img is list of path
                width,height = np.asarray(PILIM.open(img[0]).convert("L")).shape[0],np.asarray(PILIM.open(img[0]).convert("L")).shape[1]
            else: #img is list of arrays
                width,height=img[0].shape[0],img[0].shape[1]
        else:
            width,height=img.shape[0],img.shape[1]

        # Hide center from gfilter
        mask=np.ones((width,height))
        mask[border_thickness:-border_thickness,border_thickness:-border_thickness]=0
        mask_border=np.logical_not(self.create_disk_mask(mask.shape[0] - 2 * border_thickness, shape = (mask.shape[0], mask.shape[1])))

        #smaller mask
        small_gfilter=np.ones((width,height))
        border_thickness_small=border_thickness//2
        small_gfilter[border_thickness_small:-border_thickness_small,border_thickness_small:-border_thickness_small]=0

        gfilter=do_grain(mask_border,fill_value=0.01,distance_exponent=distance_exponent_big)
        small_gfilter=do_grain(small_gfilter,fill_value= 0.1,distance_exponent=distance_exponent_small)

        small_gfilter=small_gfilter.astype(bool)
        # Apply the granular gfilter on the distance transform
        gfilter = np.logical_not(gfilter).astype(np.float64)
        
        frame_list=[]
        mask_list=[]
        #Animate rotation of grain
        new_gfilter=np.zeros((width,height))

        if isinstance(img,list): #frame list
            n_frames = len(img)
            angles = np.around(np.linspace(0, 360 * nb_rotation, n_frames),2)

            for n,image in enumerate(img):
                i = angles[n]
                print("Grain anim...",int(i),"/", 360 * nb_rotation, end="\r")
                if isinstance(image,str): #img is list of path
                    image = np.asarray(PILIM.open(image).convert("L"))
                gfilter_rotated=ndimage.rotate(gfilter,i,reshape=False,cval=0,prefilter=False,mode="constant")
                new_gfilter+=gfilter_rotated-0.1*new_gfilter

                new_gfilter+=grain_fill(new_gfilter,fill_value=0.1).astype(np.float64)
                #new_gfilter %= 10

                new_gfilter = np.clip(new_gfilter,0,1.2)

                new_img=image  * new_gfilter

                mask_list.append(new_gfilter.copy())
                if self.frame_save:
                    frame_list.append((self.normalize(new_img) * 255).astype(np.uint8))
                else:
                    cv2.imwrite(self.FRAME_DIR + f"frame_{n}.png",(self.normalize(new_img) * 255).astype(np.uint8))
                    frame_list.append(self.FRAME_DIR + f"frame_{n}.png")


        else:
            angles = np.around(np.linspace(0, 360 * nb_rotation, n_frames),2)
            for n,i in enumerate(angles):
                
                if self.verbose:
                    print("Grain anim...",int(i),"/", 360 * nb_rotation, end="\r")
                gfilter_rotated=ndimage.rotate(gfilter,i,reshape=False,cval=0,prefilter=False,mode="constant")
                new_gfilter+=gfilter_rotated-0.1*new_gfilter
                

                new_gfilter+=grain_fill(new_gfilter,fill_value=0.0).astype(np.float64)
                
                
                new_gfilter = np.clip(new_gfilter,0,1.2)

                new_img=img * new_gfilter

                mask_list.append(new_gfilter.copy())
                if self.frame_save:
                    frame_list.append((self.normalize(new_img) * 255).astype(np.uint8))
                else:
                    cv2.imwrite(self.FRAME_DIR + f"frame_{n}.png",(self.normalize(new_img) * 255).astype(np.uint8))
                    frame_list.append(self.FRAME_DIR + f"frame_{n}.png")
        if self.verbose:
            print('Grain anim...Done\t\t')
        return frame_list,mask_list #int 0-255 list of array (n,n)
    
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
            if isinstance(img_obj[0],str): #img_obj is list of path
                array = np.asarray(PILIM.open(img_obj[0]).convert("L"))
                total_pixels = array.shape[0] * array.shape[1]
                width,height=array.shape[0],array.shape[1]
            else: #img_obj is list of arrays
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
                fractal_mask=self.frac_boundary[0]
                fractal_mask = np.where(fractal_mask>0.5,1,0).astype(bool)

                mask*=fractal_mask

        pulse_func = lambda x,phase: np.sin(x * 2*np.pi + phase ) + 1 #sinusoidal flicker,
        pulse_func = np.vectorize(pulse_func)

        if isinstance(img_obj,list): #multiple images
            
            for i,img_array in enumerate(img_obj):
                if self.verbose:
                    print("Flicker animation...",i,"/",len(img_obj), end="\r")
                
                if isinstance(img_array,str): #img_obj is list of path
                    img_array = np.asarray(PILIM.open(img_array).convert("L"))
                else:
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
                if self.frame_save:
                    frame_list.append(img_array)
                else:
                    cv2.imwrite(self.FRAME_DIR + f"frame_{i}.png",img_array)
                    frame_list.append(self.FRAME_DIR + f"frame_{i}.png")

               
            
        else: #single image

            # Specify the number of frames you want in the animation
            num_frames = nb_frames
            for i in range(num_frames+1):
                if self.verbose:
                    print("Flicker animation...",i,"/",num_frames, end="\r")
                
                img_array = img_obj.copy()

                # Calculate the sine value for each pixel (including phase)
                sine_values = np.sin((i / nb_frames * 2 * np.pi) + phase)

                # Apply the mask to the sine values
                sine_values_masked = sine_values * mask

                # Calculate the flicker multiplier
                flicker_multiplier = 1 + (flicker_amplitude * sine_values_masked)

                # Apply on image
                img_array = img_array * flicker_multiplier

                # Normalize the image
                img_array = (self.normalize(img_array) * 255).astype(np.uint8)

                # Add the image to the frame list
                if self.frame_save:
                    frame_list.append(img_array)
                else:
                    cv2.imwrite(self.FRAME_DIR + f"frame_{i}.png",img_array)
                    frame_list.append(self.FRAME_DIR + f"frame_{i}.png")

        if self.verbose:
            print("Flicker animation...Done\t\t\t")

        return frame_list #int 0 - 255 list of arrays(n,n)

    def Pulsing(self,img_obj,fractal_bounds,**kwargs):
        """ 
        Create a pulsing animation of the fractal image

        img_obj: single array or list of arrays
        frac_boundary: array of the fractal boundary

        kwargs
        beta: damping coefficient for oscillations
        decal: number of pixels to add to the image to avoid the animation to be cut when saving as GIF
        cmap: colormap to use for the animation
        """
        # get the parameters
        beta = kwargs.get("beta",-0.3)
        decal = kwargs.get("decal",0)
        omega = kwargs.get("oscillation_frequency",np.pi)
        amplitude = kwargs.get("oscillation_amplitude",1)
        c = kwargs.get("c",None)

        if self.verbose:
            print("Pulsing...")

        def f(u,beta = -0.03):
            return amplitude * np.exp(- beta * u) * np.sin(omega * u)
        
        if isinstance(img_obj,list):
            if isinstance(img_obj[0],str): #list of path
                img = np.asarray(PILIM.open(img_obj[0]).convert("L"),dtype=np.float64)

                frac_size = img.shape[0]
            else:
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

            if isinstance(img_obj,list):
                if isinstance(img_obj[step],str): #list of path
                    img = np.asarray(PILIM.open(img_obj[step]).convert("L"),dtype=np.float64)

                else:
                    img = img_obj[step]

            Mask = np.where(R<= c*t,1,0 ) # where c*t is peak of a Gaussian wave
            Psi = f(R - c * t,beta=beta)

            if isinstance(img_obj,list):
                try: #fractal bounds is a list
                    wave_im = (((Psi * Mask) * fractal_bounds[step] * 255)).astype(np.uint8)
                except: #fractal bounds is a list containing one array
                    wave_im = (((Psi * Mask) * fractal_bounds[0] * 255)).astype(np.uint8)

                if isinstance(img_obj[step],str):
                    img = (img * 255/np.max(img)) # image always appears

                else:
                    img = (img_obj[step] * 255/np.max(img_obj[step]))
                
                #new_im = (self.normalize(wave_im + img_obj[step]) * 255).astype(np.uint8)

            else: #single image
                wave_im = (((Psi * Mask) * fractal_bounds[0])*255).astype(np.uint8)
                
                img = (img * 255/np.max(img)) # image always appears
                
                #new_im = (self.normalize(wave_im + img) * 255).astype(np.uint8)
                
            if self.frame_save:
                frame_list.append((wave_im + img).astype(np.uint8))
            else:
                cv2.imwrite(self.FRAME_DIR + f"frame_{step}.png",(wave_im + img).astype(np.uint8))
                frame_list.append(self.FRAME_DIR + f"frame_{step}.png")
            
            if self.verbose:
                print("  ",step,"/",max_t,np.max(Psi) ,end="\r")
        if self.verbose:
            print("Pulsing...Done\t\t\t")
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
            print("(Vm-Zoom_and_Translate)...",end=" ")
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

        actual_dpi = param["Image"]["dpi"]

        zoom_speed_factor = kwargs.get("zoom_speed",1.1)

        if self.nb_frames is not None:
            self.fps = 20
            self.duration = self.nb_frames//self.fps
        nb_frames = self.fps*self.duration if self.nb_frames is None else self.nb_frames


        # we'll save the frames in a list
        frame_list = []

        # get damping list from the parameters
        if "translate" in animation:
            damping_list=np.linspace(init_damp_r,end_damp_r,self.fps*self.duration+1)+np.linspace(init_damp_c,end_damp_c,self.fps*self.duration+1)*1j
            param["Fractal"]["damping"]=damping_list[0]

        if "shading" in animation:
            nb_rotation = kwargs.get("nb_rotation",1)
            azimuth = np.linspace(0,22.5 * nb_rotation, nb_frames)


        def check_coord(z,edges,dpi,coord,zoom,prev_point):
            #init cartesian coord
            array=self.init_array(dpi,coord)

            #init distance map to the center
            dist_map = np.ones_like(z)
            dist_map[z.shape[0]//2,z.shape[1]//2] = 0
            dist_map = ndimage.distance_transform_edt(dist_map)

            #set distance map on edge only, other values are set to inf
            dist_map = np.where(edges>0,dist_map,np.nan)
            #plot
            #get the point with the min distance to the center
            point = np.unravel_index(np.nanargmin(dist_map), dist_map.shape)

            if np.any(point) == False:
                return coord*zoom,prev_point

            #get the point value in z
            pts = array[point[0],point[1]]
            pts = [pts.real,pts.imag]

            #update coord
            coord=np.array([[(pts[0])-1*zoom,(pts[0])+1*zoom], #real
                        [(pts[1])-1*zoom,(pts[1])+1*zoom]]) #complex
            
            return coord,point

        #loop over frames
        zoom_speed = 1
        zoom_speed_for_border = 1
        param["Fractal"]["form"]=None
        param["Fractal"]["pts"] = [0,0]
        for _ in range(nb_frames):
        # num_frames is self.fps*self.duration
            print("\t\t\t\t\t\t",end="\r")
            print("Zoom and Translate and Shading",_,"\t",end="\r")
            #Create frame

            #zoom logic
            if _  == 0 :
                if "zoom" in animation and "translate" not in animation:
                    param["Image"]["dpi"] = 500
                    param["Fractal"]["size"] = 500
                Imobj, im = self.IMAGE_wrapper_for_fractal(param)
                
                if "zoom" in animation and "translate" not in animation:
                    Imobj.frac_boundary = np.asarray(PILIM.fromarray(Imobj.frac_boundary.astype(np.uint8)).resize((500,500),resample=PILIM.BICUBIC).convert("L"))
                    Imobj.frac_boundary = np.where(Imobj.frac_boundary>0.5,1,0).astype(bool)

                param["Fractal"]["func"]= Imobj.func
                param["func"] = Imobj.func
                param["Fractal"]["random"]=False #True only for first frame at most
                param["random"] = False
                param["Fractal"]["verbose"]=False
                param["Image"]["verbose"]=False
                param["verbose"] = False

            else:
                zoom_speed = zoom_speed/zoom_speed_factor
                zoom_speed_for_border = zoom_speed_for_border/zoom_speed_factor
            
            if ("zoom" in animation) and ("translate" not in animation) and (_ % self.fps == 0): # create bigger image, that we can use to zoom in without recalculation each frames

                param["Image"]["dpi"] = 5000
                param["Fractal"]["size"] = 5000
                if _ == 0:
                    param["Fractal"]["domain"],param["Fractal"]["pts"]=check_coord(im,Imobj.frac_boundary,500 if _ ==0 else 5000,param["Fractal"]["domain"],zoom_speed,prev_point=param["Fractal"]["pts"])

                    Imobj, big_im = self.IMAGE_wrapper_for_fractal(param)
                    big_im = big_im.copy().astype(np.uint8)

                else:
                    Imobj, big_im = self.IMAGE_wrapper_for_fractal(param)
                    big_im = big_im.copy().astype(np.uint8)
                    boundary_temp = np.asarray(PILIM.fromarray(Imobj.frac_boundary.astype(np.uint8)).resize((5000,5000),resample=PILIM.BICUBIC).convert("L"))
                    param["Fractal"]["domain"],param["Fractal"]["pts"]=check_coord(big_im,boundary_temp,500 if _ ==0 else 5000,param["Fractal"]["domain"],zoom_speed,prev_point=param["Fractal"]["pts"])

                #resize big_im
                im = np.asarray(PILIM.fromarray(big_im).resize((actual_dpi,actual_dpi),resample=PILIM.BICUBIC))
                param["Image"]["dpi"] = 2000
                param["Fractal"]["size"] = 2000

                #augment tolerance after 60 frames
                if  _ > 60:
                    param["Fractal"]["tol"]  = param["Fractal"]["tol"] * 0.1

                zoom_speed_for_border = 1
            
            elif "zoom" in animation and "translate" not in animation:
                #zoom in on the image (cut the borders)
                        # Calculate cropping borders
                x1 = int(5000 * (1 - zoom_speed_for_border) / 2)
                y1 = int(5000 * (1 - zoom_speed_for_border) / 2)
                x2 = int(5000 * (1 + zoom_speed_for_border) / 2)
                y2 = int(5000 * (1 + zoom_speed_for_border) / 2)
                im = big_im[y1:y2,x1:x2]
                im = np.asarray(PILIM.fromarray(im.astype(np.uint8)).resize((actual_dpi,actual_dpi),resample=PILIM.BICUBIC))

            elif "zoom" in animation and "translate" in animation:
                
                param["Fractal"]["domain"],param["Fractal"]["pts"]=check_coord(im,Imobj.frac_boundary,param["Image"]["dpi"],param["Fractal"]["domain"],zoom_speed,prev_point=param["Fractal"]["pts"])
                Imobj, im = self.IMAGE_wrapper_for_fractal(param)
                im = im.copy().astype(np.uint8)

                #augment tolerance after 60 frames
                if  _ > 60:
                    param["Fractal"]["tol"]  = param["Fractal"]["tol"] * 0.01
                
            #translate logic
            if "translate" in animation:
                if "zoom" not in animation:
                    Imobj, im = self.IMAGE_wrapper_for_fractal(param)
                param["Fractal"]["damping"]=damping_list[_]

            #save frames
            if "zoom" in animation or "translate" in animation:
                self.frac_boundary.append(cv2.resize(Imobj.frac_boundary.astype(np.uint8),(actual_dpi,actual_dpi),interpolation=cv2.INTER_NEAREST))

            if "shading" in animation:
                shade_im = self.Dynamic_shading(Imobj,azimuth = [azimuth[_]] ,**kwargs)[0]

                if "zoom" in animation or "translate" in animation:
                    pass
                else:
                    self.frac_boundary = cv2.resize(Imobj.frac_boundary.astype(np.uint8),(actual_dpi,actual_dpi),interpolation=cv2.INTER_NEAREST)

                if self.frame_save:
                    frame_list.append((self.normalize(shade_im) * 255).astype(np.uint8))
                else:
                    cv2.imwrite(self.FRAME_DIR + f"frame_{_}.png",(self.normalize(shade_im) * 255).astype(np.uint8))
                    frame_list.append(self.FRAME_DIR + f"frame_{_}.png")
            else: #no shading
                if self.frame_save:
                    frame_list.append((self.normalize(im) * 255).astype(np.uint8))
                else:
                    cv2.imwrite(self.FRAME_DIR + f"frame_{_}.png",(self.normalize(im) * 255).astype(np.uint8))
                    frame_list.append(self.FRAME_DIR + f"frame_{_}.png")

        if self.verbose:
            print("Zoom and Translate and Shading...Done\t\t\t")

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
            print("(Vm-Dynamic_shading)...",end="\r")
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
            if self.frame_save or len(azimuth_list) == 1: # if only one frame, return array
                frame_list.append((self.normalize(shade) * 255).astype(np.uint8))
            else:
                cv2.imwrite(self.FRAME_DIR + f"frame_{azimuth}.png",(self.normalize(shade) * 255).astype(np.uint8))
                frame_list.append(self.FRAME_DIR + f"frame_{azimuth}.png")

        if self.verbose:
            print("(Vm-Dynamic_shading)...Done",end="\r")

        return frame_list

