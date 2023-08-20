import os
import sys
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image as PILIM

import cv2
from scipy import ndimage

from skimage.filters import threshold_mean
from skimage.feature import canny
from skimage.morphology import disk,dilation

import time
from Image import IMAGE

def clean_dir(folder):
    import shutil
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
        self.IM_DIR=os.path.join(self.APP_DIR,"images")
        try: os.mkdir(self.IM_DIR)
        except: pass
        self.VID_DIR=os.path.join(self.APP_DIR,"video")
        try: os.mkdir(self.VID_DIR)
        except: pass

        self.FRAME_DIR=os.path.join(self.VID_DIR,"frames")
        try: os.mkdir(self.FRAME_DIR)
        except: pass
        clean_dir(self.FRAME_DIR)

        ### Set paramaters
        self.set_video_parameters(param)

        print("Done (V-Init)")

    def set_video_parameters(self,param):
        self.fps=param["fps"]
        self.seconds=param["duration"]

        self.frame_name=0


        self.zoom=1
        self.zoom_speed=param["zoom"]

        self.frac_boundary = []


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
    
    ### VIDEO MAKER ###
    def Video_maker(self,param):
        print("Video maker (V-vm)...",end="")

        #Copy frame dir to Desktop
        from time import strftime,gmtime
        import shutil
        dt_gmt = strftime("%Y_%m_%d_%H_%M_%S", gmtime())

        src_dir=self.FRAME_DIR
        dest_dir=os.path.join(self.APP_DIR,f"video/video_{dt_gmt}")
        shutil.copytree(src_dir, dest_dir)

        #Create video
        #self.Create_video(dest_dir,param["FPS"])
        
        print("Done (V-vm)")

    ### ANIMATIONS ###
    def Grain_anim(self,im_path,im_path_2=None, **kwargs):
        '''Generate explosion animation of input image, with rotational grain effect and flickering
        im_path: path of image to explode
        im_path_2 (optionnal): put explosion on top of this image
        '''
        def deteriorate_border_anim(img, border_thickness=200, hole_size=3, n_frames=100):
            """
            Apply granular gfilter to an image

            img: a numpy array of shape (height, width, 3)
            border_thickness: the thickness of the granular gfilter from the border (pixels)
            hole_size: the size of the holes (radius of the disk structuring element)

            return: a numpy array of shape (height, width, 3)
            """
            print("deterioating border...",end="")

            def do_grain(mask_border,fill_value=0.3):
                """
                grainification of an image

                """
                # Apply distance transform
                dist_transform = cv2.distanceTransform(mask_border.astype(np.uint8), cv2.DIST_L2, 5)
                # Normalize the distance transform image for viewing
                mask = (cv2.normalize(dist_transform, None, alpha=fill_value, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
                # create the granular gfilter
                gfilter = np.zeros(mask.shape)
                gfilter = np.random.rand(*mask.shape) <= mask
                return gfilter

            def grain_fill(gfilter,fill_value=0.3):
                """
                fill empty space in  gfilter with more grain
                """
                dilation=ndimage.binary_fill_holes(gfilter.copy())

                # Apply distance transform
                dist_transform = np.float_power(cv2.distanceTransform(dilation.astype(np.uint8), cv2.DIST_LABEL_PIXEL, 5),0.7)
                # Normalize the distance transform image for viewing
                mask = (cv2.normalize(dist_transform, None, alpha=fill_value, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
                # create the granular gfilter
                new_filter = np.zeros(mask.shape)
                new_filter = np.random.rand(*mask.shape) <= mask

                return new_filter
                
            width,height=img.shape[0],img.shape[1]

            # Hide center from gfilter
            mask=np.ones((width,height))
            mask[border_thickness:-border_thickness,border_thickness:-border_thickness]=0
            mask_border=np.copy(mask)

            #smaller mask
            small_gfilter=np.ones((width,height))
            border_thickness_small=border_thickness//2
            small_gfilter[border_thickness_small:-border_thickness_small,border_thickness_small:-border_thickness_small]=0

            gfilter=do_grain(mask_border,fill_value=0.0)
            small_gfilter=do_grain(small_gfilter,fill_value=0.3)

            small_gfilter=small_gfilter.astype(bool)
            # Apply the granular gfilter on the distance transform
            gfilter[small_gfilter]= ndimage.binary_dilation(gfilter[small_gfilter],iterations=hole_size)
            gfilter = np.logical_not(gfilter).astype(np.float64)
            
            frame_list=[]
            
            #Animate rotation of grain
            new_gfilter=np.zeros((width,height))

            if isinstance(img,list): #frame list
                n_frames = len(img)
                angles = np.around(np.linspace(0, 360, n_frames),2)

                for image in img:
                    i = angles[img.index(image)]
                    print("rotation grain anim: ",int(i),"/360", end="\r")
                    gfilter_rotated=ndimage.rotate(gfilter,i,reshape=False,cval=0,prefilter=False,mode="constant")
                    new_gfilter+=gfilter_rotated-0.3*new_gfilter

                    new_gfilter+=grain_fill(new_gfilter,fill_value=0.0).astype(np.float64)
                    #new_gfilter %= 10

                    new_gfilter = np.clip(new_gfilter,0,1.2)


                    # Convert the numpy array back to a PIL Image and append it to the frames list
                    try:
                        new_img=image * np.repeat(new_gfilter[:, :, np.newaxis], 4, axis=2)
                        print("RGBA mode ", end="")
                    except:
                        try:
                            new_img=image * np.repeat(new_gfilter[:, :, np.newaxis], 3, axis=2)
                            print("RGB mode ", end="")
                        except:
                            new_img=image * new_gfilter
                            print("L mode ",end="")
                
                    frame_list.append(np.uint8(new_img)) # to save as gif


            else:
                angles = np.around(np.linspace(0, 360, n_frames),2)
                for i in angles:

                    print("rotation grain anim: ",int(i),"/360", end="\r")
                    gfilter_rotated=ndimage.rotate(gfilter,i,reshape=False,cval=0,prefilter=False,mode="constant")
                    new_gfilter+=gfilter_rotated-0.3*new_gfilter

                    new_gfilter+=grain_fill(new_gfilter,fill_value=0.0).astype(np.float64)
                    #new_gfilter %= 10

                    new_gfilter = np.clip(new_gfilter,0,1.2)


                    # Convert the numpy array back to a PIL Image and append it to the frames list
                    try:
                        new_img=img * np.repeat(new_gfilter[:, :, np.newaxis], 4, axis=2)
                        print("RGBA mode ", end="")
                    except:
                        try:
                            new_img=img * np.repeat(new_gfilter[:, :, np.newaxis], 3, axis=2)
                            print("RGB mode ", end="")
                        except:
                            new_img=img * new_gfilter
                            print("L mode ",end="")
                
                    frame_list.append(np.uint8(new_img)) # to save as gif

            print('deterioate border anim done')
            return frame_list
    
        #explosion
        self.EX_DIR=os.path.join(self.VID_DIR,"explosion")
        clean_dir(self.EX_DIR)
        try:
            os.mkdir(self.EX_DIR)
        except:
            pass

        image=PILIM.open(im_path)

        #Explosion is resizing of image w/ filter
        inf_size=(1,1)
        sup_size=(image.size[0],image.size[1])
        list_ex=((np.arange(inf_size[0],np.sqrt(sup_size[0])))**2).astype(int)
        list_ex=np.append(np.unique(list_ex),sup_size[0])

        #Create frames
        frame_list = deteriorate_border_anim(image,**kwargs)
        
        # Apply resizing explosion
            #match size of explosion to size of image
        for i in range(len(frame_list)-len(list_ex)):
            list_ex = np.append(list_ex,list_ex[-1])
        for i in range(len(frame_list)):
            frame_list[i]=frame_list[i].resize((list_ex[i],list_ex[i]))
        
        return frame_list
    
    def Flicker(self,img_input,flicker_percentage=0.0005,on_fractal=False, dilation_size = 2,flicker_amplitude=0.9, n_frames=300):
        """
        Apply a flicker animation to an image

        img: a numpy array of shape (height, width, 3), or a list of such arrays
        
        flicker_amplitude: the amplitude of the flicker effect 
        flicker_percentage: the percentage of pixels to flicker
        n_frames: the number of frames in the animation (single image only)
        on_fractal: whether to apply the flicker on the fractal or everywhere

        return: a list of numpy arrays of shape [frames,(height, width, 3)]
        """
        print("Flicker animation...", end="")
        # We'll store each frame as we create it
        frames = []
        # Calculate the total number of pixels
        if isinstance(img_input,list):
            total_pixels = img_input[0].shape[0] * img_input[0].shape[1]
            width,height=img_input[0].shape[0],img_input[0].shape[1]
        else: 
            total_pixels = img_input.shape[0] * img_input.shape[1]
            width,height=img_input.shape[0],img_input.shape[1]
        num_flicker_pixels = int(total_pixels * flicker_percentage)

        # Generate the indices of the pixels to flicker
        flicker_indices = np.random.choice(total_pixels, num_flicker_pixels, replace=False)

        #Create mask from flicker indices
        mask=np.zeros((width,height))
        np.put(mask,flicker_indices,1)
        mask=(mask).astype(bool)    

        #assign random values to flicker indices
        mask_start=np.zeros((width,height))
        np.put(mask_start,flicker_indices,np.random.uniform(0,255,num_flicker_pixels))

        #dilate mask
        for i in range(dilation_size):
            
            mask_start+=dilation(mask_start,disk(i))
        mask=ndimage.binary_dilation(mask,iterations=dilation_size)

        # Apply flicker effect on fractal only, if specified
        if on_fractal:

            fractal_mask=np.asarray(PILIM.open("images/test0_nobckg.png").convert("L"))/255
            fractal_mask = np.where(fractal_mask>0.5,1,0)

            mask_start*=fractal_mask

        # Create a function that will be used to generate the flicker effect
        def smooth_step(x,a,b):
            x=np.where((x>=a)&(x<=b),3*(x-a)**2/(b-a)**2-2*(x-a)**3/(b-a)**3,np.where(x<a,0,1))
            return x
        
        pulse_func=lambda x: (smooth_step(np.mod(x,1.5),-4,0.6)-smooth_step(np.mod(x,1.5),-5-1/3,0.8))/0.006821  #experimentally determined (lol)

        if isinstance(img_input,list): #multiple images
            i = 0
            for img_array in img_input:
                img_array = img_array.copy()
                # Create a multiplier that goes between 1-flicker_amplitude and 1+flicker_amplitude
                # We use a function to create a smooth flicker effect
                flicker_multiplier = np.array((1 + flicker_amplitude * pulse_func((mask_start * i) + 1.5)))[mask]

                if img_array.shape[-1]==4:
                    #RGBA
                    flicker_multiplier = np.repeat(flicker_multiplier[:, np.newaxis], 4, axis=1)
                elif img_array.shape[-1]==3:
                    #RGB
                    flicker_multiplier = np.repeat(flicker_multiplier[:, np.newaxis], 3, axis=1)
                else:
                    #L
                    pass

                # apply the flicker effect
                img_array[mask] = np.clip(img_array[mask] * flicker_multiplier,0,200)

                # Convert the numpy array back to a PIL Image and append it to the frames list
                frames.append(np.uint8(img_array))

                i+=1/len(img_input)
            
        else: #single image

            # Specify the number of frames you want in the animation
            num_frames = n_frames
            for i in range(num_frames+1):
                # Create a multiplier that goes between 1-flicker_amplitude and 1+flicker_amplitude
                # We use a function to create a smooth flicker effect
                flicker_multiplier = np.array((1 + flicker_amplitude * pulse_func(mask_start * i / num_frames+1.5)))[mask]

                if img_input.shape[-1]==4:
                    #RGBA
                    flicker_multiplier = np.repeat(flicker_multiplier[:, np.newaxis], 4, axis=1)
                elif img_input.shape[-1]==3:
                    #RGB
                    flicker_multiplier = np.repeat(flicker_multiplier[:, np.newaxis], 3, axis=1)
                else:
                    #L
                    pass
                # copy the array so we don't overwrite the original
                img_array = img_input.copy()

                # apply the flicker effect
                img_array[mask] = np.clip(img_array[mask] * flicker_multiplier,0,200)

                # Convert the numpy array back to a PIL Image and append it to the frames list
                frames.append(np.uint8(img_array))

        print("flicker anim done")

        return frames

    def Pulsing(img_obj,fractal_bounds,beta=None,decal=0,**kwargs):
        """ 
        Create a pulsing animation of the fractal image

        img_obj: single array or list of arrays
        frac_boundary: array of the fractal boundary, if None, it is img_obj.frac_boundary (img_obj must be an IMAGE object)
        """
        print("Pulsing...")
        def f(u,beta = -0.03):
            #beta = -0.03  # Damping coefficient for oscillations
            omega = 2*np.pi # Frequency of oscillations
            A = 1 #amplitude
            return A * np.exp(- beta * u) * np.sin(omega * u)
        
        if isinstance(img_obj,list):
            img = img_obj[0]

            frac_size = img.shape[0]
            # Time
            max_t = len(img_obj)

            #speed of the wave
            c = (frac_size + decal)//max_t
        else:
            img = img_obj

            frac_size = img_obj.shape[0]
            
            c = frac_size // 300  # speed of the wave
            # Time
            max_t = (frac_size + decal) // c
        
        if beta is None:
            beta = - 25 / frac_size
        
        # Gif properties
        # Source location
        x_center, y_center = frac_size // 2, frac_size // 2
        # Wave properties



        # frame array
        frame_array = []

        X, Y = np.meshgrid(np.arange(frac_size), np.arange(frac_size))
        R = np.sqrt((X - x_center)**2 + (Y - y_center)**2) #distance

        Mask = np.zeros((frac_size, frac_size))
        
        for step,t in enumerate(np.arange(0, max_t, 1)):

            Mask = np.where(R<= c*t,1,0 ) # where c*t is peak of a Gaussian wave
            Psi = f(R - c * t,beta=beta)

            if isinstance(img_obj,list):
                wave_im = (((Psi * Mask) * fractal_bounds[step] * 255)).astype(np.uint8)

                img_obj[step] = (img_obj[step] * 255/np.max(img_obj[step]))
                plt.imsave(f"images/pulse.png", wave_im + img_obj[step]//3, vmin=0, vmax=255, **kwargs)
            else: #single image
                wave_im = (((Psi * Mask) * fractal_bounds * 255)).astype(np.uint8)
                
                img = (img * 255/np.max(img)) # image always appears
                #or
                #img /= np.max(img) # image appears with wave
                plt.imsave(f"images/pulse.png", wave_im + img//2, vmin=0, vmax=255, **kwargs)
            
            wave_im = PILIM.open(f"images/pulse.png").resize(img.shape)
            wave_im = np.asarray(wave_im)

            frame_array.append(wave_im)
            print("  ",step,"/",max_t,np.max(Psi) ,end="\r")
        print("Pulsing done")
        return frame_array
    #Zoom
    def Zoom(self,param):
        frame_list = []
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
        
        start_time =  time.time()
        for _ in range(self.fps*self.seconds):
            print("  ",_,end="")
            #Create frame
            Imobj=IMAGE(param) 
            im = Imobj.Fractal_image()

            param["func"]= Imobj.func
            param["form"]=None
            param["random"]=False #True only for first frame at most
            param["verbose"]=False
            param["pts"] = [0,0]
            #save frame
            self.frac_boundary.append(Imobj.frac_boundary)
            frame_list.append(im)

            # update zoom
            self.zoom = self.zoom/self.zoom_speed
            param["domain"],param["pts"]=check_coord(im,Imobj.frac_boundary,param["dpi"],param["domain"],self.zoom,prev_point=param["pts"])

            end_time = time.time()
            if _ == 0:
                total_time = (end_time-start_time)*self.fps*self.seconds
            print("  progress: ",np.around((end_time-start_time),2),"/",total_time, "time left: ",np.around((total_time - np.around((end_time-start_time),2))/60,2),"min",end = "\r")
            #param["domain"]=param["domain"]*self.zoom
        return frame_list

    def Translate(self, param, init_damp_r = 0.4, end_damp_r = 1.35, init_damp_c = -0.5, end_damp_c = 0.85):
        frame_list = []
        damping_list=np.linspace(init_damp_r,end_damp_r,self.fps*self.seconds+1)+np.linspace(init_damp_c,end_damp_c,self.fps*self.seconds+1)*1j
        param["damping"]=damping_list[0]

        for _ in range(self.fps*self.seconds):
            print(_,end="\r")

            #Create frame
            Imobj=IMAGE(param) 
            im = Imobj.Fractal_image(param)
            param["func"]=Imobj.func

            param["random"] = False
            param["damping"]=damping_list[_]
            #save frame
            frame_list.append(im)

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
        obj=Videos(param)
        obj.Video_maker(param)

    except KeyboardInterrupt:
        sys.exit()
        