import os
import sys

import random
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image as PILIM

from scipy.ndimage import gaussian_filter,sobel,binary_dilation
from skimage.filters import threshold_otsu,threshold_local
from skimage.feature import canny

from RFA_fractals import RFA_fractal
### GLOBAL FONCTIONS ###
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

class IMAGE():

    def __init__(self,parameters) -> None:
        print("Init Image class (IM-Init)...")
        ### Create folders
        self.APP_DIR = os.path.dirname(os.path.abspath(__file__))
        self.IM_DIR=os.path.join(self.APP_DIR,"images")
        self.FRAC_DIR=os.path.join(self.IM_DIR,"fractal")
        try: os.mkdir(self.IM_DIR)
        except: pass
        try: os.mkdir(self.FRAC_DIR)
        except: pass

        ### Set paramaters
        self.dpi=parameters["dpi"]
        self.cmap=parameters["cmap"]

        print("Done (IM-Init)")
    
    ### PLOT ###
    def Plot(self,array,name,Dir,print_create=True):
        """Plot a numpy array as an image"""
        fig = plt.figure(frameon=False)
        fig.set_size_inches(1,1)    
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(array,cmap=self.cmap)

        fig.savefig(Dir+"/"+name,dpi=self.dpi) 
        if print_create==True:
            print("created figure '% s'" %name,end="")
            print("in folder '% s'" %Dir)
        plt.close(fig)
        return None
    
    ### FRACTAL IMAGE ###
    def Fractal_image(self, parameters):
        print("Fractal_image (IM-Fim)...")

        parameters["dir"]=self.FRAC_DIR
        frac_param={
            "N":300,
            "domain":np.array([[-1,1],[-1,1]]),

            "random":True,
            "func":[1,-1/2+np.sqrt(3)/2*1j,-1/2-np.sqrt(3)/2*1j],
            "form": "root",

            "method":"Nova Newton",

        }

        frac_obj=RFA_fractal(frac_param)
        frac_param["func"]=frac_obj.coefs

        if "Nova" in frac_param["method"]:
            c=frac_obj.array #corresponding pixel to complex plane
            shape=c.shape
            c=c.flatten()

            c_coefs=frac_obj.poly.add_c_to_coefs(c,frac_param["func"],frac_param["random"])

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
                self.z,conv=frac_obj.Nova_Newton_method(d2roots, #z0
                                                        lambda z: np.array([frac_obj.poly.poly(z[i],c_coefs[i]) for i in range(len(c_coefs))]), #f
                                                        lambda z: np.array([frac_obj.poly.dpoly(z[i],c_coefs[i]) for i in range(len(c_coefs))]), #f'
                                                        tol=1.e-05,
                                                        max_steps=50,
                                                        damping=complex(1,0.2))


                self.z,conv=self.z.reshape(shape),conv.reshape(shape)

                self.z,conv=self.z.real,conv.real

                #Binary map
                frac_entire=self.Local_treshold(conv)         
       
        
        else:
            if "Newton" in frac_param["method"]:                
                self.z,conv=frac_obj.Newton_method(frac_obj.array,
                                                    lambda z: frac_obj.poly.poly(z,frac_obj.coefs),
                                                    lambda z: frac_obj.poly.dpoly(z,frac_obj.coefs),
                                                    1.e-05,
                                                    50,
                                                    complex(1,0.2))
                self.z,conv=self.z.real,conv.real

                #Binary map
                #frac_entire=self.Local_treshold(conv) 
                frac_entire=binary_dilation((canny(conv)+self.Local_treshold(conv*(-1)) +sobel(conv)),iterations=2)

        self.file_name="fractal_array"
        parameters["dir"]=self.IM_DIR+"/fractal"

        # Save images
        self.Plot(self.z,self.file_name,parameters["dir"])
        self.Plot(conv,"convergence",parameters["dir"])
        self.Plot(frac_entire,"edges",parameters["dir"])

        print("Done (IM-Fim)")

        return conv
    
    def Orbit_trap(self,parameters):
        pass

    def adaptive_antialiasing(self,parameters):
        pass
    ### IMAGE HANDLER ###
    def crop(self,im_path):
        image=PILIM.open(im_path)

        imageBox = image.getbbox()
        cropped = image.crop(imageBox)
        cropped.save(im_path)

    def paste_image(self,image_bg, image_element, x0, y0, x1, y1, rotate=0, h_flip=False):
        """
            image_bg: image in the background
            image_element: image that will be added on top of image_bg
            x0,y0,y1,x1: coordinates of the image_element on image_bg
            rotate (0 if None): rotation of image_element
            h_flip (False by default): horizontal flip of image_element
        """
        #Copy all images
        image_bg_copy = image_bg.copy()
        image_element_copy = image_element.copy()
        image_element_copy=image_element_copy.resize((x1-x0,y1-y0))


        #horizontal flip
        if h_flip:
            image_element_copy = image_element_copy.transpose(PILIM.FLIP_LEFT_RIGHT)

        #do rotation
        image_element_copy = image_element_copy.rotate(rotate, expand=True)

        #get all chanel
        _, _, _, alpha = image_element_copy.split()

        # image_element_copy's width and height will change after rotation
        w = image_element_copy.width
        h = image_element_copy.height

        #Final image
        image_bg_copy.paste(image_element_copy, box=(x0, y0, x1, y1), mask=alpha)
        return image_bg_copy


    ############RENDERING#################
    ### COLORS ###

    ### FILTERS ###
    def Local_treshold(self,array):
        """Local treshold filter"""
        return array>threshold_local(array)



if __name__=='__main__':
    parameters={
    "frame_name":1,
    "Animation":"Pulsing",
    "Repetition":1,
    "FPS":30 ,
    "Duration":5, #seconds
    "zoom":1.2,
    "color_API_tuple":[[64,15,88],(15,5,0),(249,44,6),"N","N"],  #Example: [[r,g,b],"N","N","N","N"]
    "cmap":"Greys",
    "coord":np.array([[-1,1],[-1,1]]),
    "degree": random.randint(5,20),
    "rand_coef": False,
    "coef": [1j,1,1j,1j,1j,0,0], #Must have value if rand_coef==False
    "dpi": 1000,
    "itermax":150,
    # RFA
}

    try:
        obj=IMAGE(parameters)
        obj.Fractal_image(parameters)

    except KeyboardInterrupt:
        sys.exit()