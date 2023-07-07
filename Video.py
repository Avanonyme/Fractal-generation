import os
import sys
import numpy as np

import matplotlib
matplotlib.use('agg')
from PIL import Image as PILIM

import cv2
from scipy import ndimage

from skimage.filters import threshold_mean
from skimage.feature import canny
from skimage.morphology import disk,dilation


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


class Videos():
    def __init__(self,param) -> None:
        print("Init Videos class(V-Init) fractals...")

        ### Create folders
        self.APP_DIR = os.path.dirname(os.path.abspath(__file__))
        self.IM_DIR=os.path.join(self.APP_DIR,"images")
        self.VID_DIR=os.path.join(self.APP_DIR,"video")
        try: os.mkdir(self.IM_DIR)
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

        if param["anim method"]=='zoom':
            self.zoom=1
            self.zoom_speed=param["zoom"]
        elif param["anim method"]=="water move":
            pass

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
    #### TRANSLATION ####
    #RFA
    def translate_poly(self,coords):
        """
        z: array of complex numbers
        coords: list coef of polynomial
        """

        #np.array([[-1,1],[-1,1]]) format
        coords[0,:]+=1e-2 #translation on real axis
        return coords

    #### ZOOM ####
    def check_coord(self,z,dpi,coord,zoom):
        print("RFA-checking coord...")

        #check for edges 
        bitmap=z > threshold_mean(z)

        #init old coord
        array=self.init_array(dpi,coord)
        print("array extremity",array[0,0],array[-1,-1])

        edges=canny(bitmap,0.1)


        for i in range(10,int(z.size),5):
            mask=self.circle_mask(z,dpi,i)
            points=np.where(edges==mask,mask,np.zeros_like(mask))
            if np.any(points)==True:
                candidate=np.where(points==True)
                point=[candidate[0][0],candidate[1][0]]
                print("Array indice",point)
                break
        try:
            pts=array[point[0],point[1]]
        except:
            return coord*zoom,[0,0]
        pts=[pts.real,pts.imag]

        coord=np.array([[(pts[0])-1*zoom,(pts[0])+1*zoom], #real
                    [(pts[1])-1*zoom,(pts[1])+1*zoom]]) #complex
        print("Done (RFA-checking coord)")
        return coord,point
    
    ### VIDEO MAKER ###
    def Video_maker(self,param):
        print("Video maker (V-vm)...",end="")
        if param["anim method"]=="rotation":
            damping_list=np.linspace(0.4,1.35,self.fps*self.seconds+1)+np.linspace(-0.5,0.85,self.fps*self.seconds+1)*1j
            param["damping"]=damping_list[0]

        if param["anim method"]=="grain": #in grain animation, we only need one image to create the video
            ...
        else: 
            #Create video
            for _ in range(self.fps*self.seconds):
                print(_,end="\r")

                #Create frame
                Imobj=IMAGE(param) 
                Imobj.Fractal_image(param)
                param["func"]=Imobj.func

                #Save frame
                PILIM.open(os.path.join(param["dir"],Imobj.file_name)+".png").save(self.FRAME_DIR+f"/frame_{self.frame_name}.png")

                #update frame
                self.frame_name+=1

                ## Update parameters
                #RFA Fractal
                param["random"]=False #True only for first frame at most

                if param["anim method"]=="zoom":
                    #Update zoom
                    self.zoom=self.zoom/self.zoom_speed
                    param["domain"],param["pts"]=self.check_coord(self.z,self.dpi,param["domain"],self.zoom)

                elif param["anim method"]=="rotation":
                    param["damping"]=damping_list[self.frame_name]
                    print("damping",param["damping"])


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
    def deteriorate_border_anim(self,img, border_thickness=200, hole_size=3, n_frames=100):
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
        
        new_img_list=[]
        
        #Animate rotation of grain
        new_gfilter=np.zeros((width,height))
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
        
            new_img_list.append(np.uint8(new_img)) # to save as gif

        print('deterioate border anim done')
        return new_img_list
    
    def flicker(img,flicker_percentage=0.0005,on_fractal=False, dilation_size = 2,flicker_amplitude=0.9, n_frames=100):
        """
        Apply a flicker animation to an image

        img: a numpy array of shape (height, width, 3), or a list of such arrays
        
        flicker_amplitude: the amplitude of the flicker effect 
        flicker_percentage: the percentage of pixels to flicker
        n_frames: the number of frames in the animation (single image only)
        on_fractal: whether to apply the flicker on the fractal or everywhere

        return: a list of numpy arrays of shape [frames,(height, width, 3)]
        """
        # We'll store each frame as we create it
        frames = []

        # Calculate the total number of pixels
        if isinstance(img,list):
            total_pixels = img[0].shape[0] * img[0].shape[1]
            width,height=img[0].shape[0],img[0].shape[1]
        else: 
            total_pixels = img.shape[0] * img.shape[1]
            width,height=img.shape[0],img.shape[1]
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

        if isinstance(img,list): #multiple images
            i = 0
            for img_array in img:

                # Create a multiplier that goes between 1-flicker_amplitude and 1+flicker_amplitude
                # We use a function to create a smooth flicker effect
                flicker_multiplier = np.array((1 + flicker_amplitude * pulse_func((mask_start * i)/300 + 1.5)))[mask]

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

                i+=1/len(img)
            
        else: #single image

            # Specify the number of frames you want in the animation
            num_frames = n_frames
            for i in range(num_frames+1):
                # Create a multiplier that goes between 1-flicker_amplitude and 1+flicker_amplitude
                # We use a function to create a smooth flicker effect
                flicker_multiplier = np.array((1 + flicker_amplitude * pulse_func(mask_start * i / num_frames+1.5)))[mask]

                if img.shape[-1]==4:
                    #RGBA
                    flicker_multiplier = np.repeat(flicker_multiplier[:, np.newaxis], 4, axis=1)
                elif img.shape[-1]==3:
                    #RGB
                    flicker_multiplier = np.repeat(flicker_multiplier[:, np.newaxis], 3, axis=1)
                else:
                    #L
                    pass
                # copy the array so we don't overwrite the original
                img_array = img.copy()

                # apply the flicker effect
                img_array[mask] = np.clip(img_array[mask] * flicker_multiplier,0,200)

                # Convert the numpy array back to a PIL Image and append it to the frames list
                frames.append(np.uint8(img_array))

        print("flicker anim done")

        return frames

    def Pulsing(self,images,on_fractal = False):
        '''Generate pulsing animation in input images
        images (list or str): list of image paths, or path of directory containing images
        '''        
        #pulsing



    def Grain_anim(self,im_path,im_path_2=None):
        '''Generate explosion animation of input image, with rotational grain effect and flickering
        im_path: path of image to explode
        im_path_2 (optionnal): put explosion on top of this image
        '''
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
        for i in list_ex:
            print(i,end="\r")
            image=PILIM.open(im_path).resize((i,i))

            # apply filter
            image=self.deteriorate_border(image)

            #save image
            image.save(os.path.join(self.EX_DIR,f"explosion_{i}.png"))

        ...
        #grain

        #flicker

        #Create video
        self.Create_video(self.EX_DIR,self.fps)

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
        