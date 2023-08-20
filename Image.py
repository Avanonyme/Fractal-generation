import os
import sys

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from PIL import Image as PILIM

from scipy.ndimage import gaussian_filter,sobel,binary_dilation
from skimage.filters import threshold_local
from skimage.feature import canny

from RFA_fractals import RFA_fractal

### GLOBAL FONCTIONS ###
def clean_dir(folder,verbose=False):
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

class IMAGE():
    ### SET PARAM^ ###
    def __init__(self,param) -> None:
        if param["verbose"]:
            print("Init Image class...",end="\r")
        ### Create folders
        self.APP_DIR = os.path.dirname(os.path.abspath(__file__))
        self.IM_DIR=os.path.join(self.APP_DIR,param["dir"])

        try: os.mkdir(self.IM_DIR)
        except: pass#print(f"Could not create folder {self.IM_DIR}" )

        ### Set paramaters
        self.set_image_parameters(param)
        self.print=param["verbose"]

        if param["verbose"]:
            print("Init Image class...Done")
    
    def set_image_parameters(self,param):
        self.param=param
        self.dpi=param["dpi"]
        self.file_name=param["file_name"]
        
        if param["clean_dir"]:
            clean_dir(self.IM_DIR)
        

        self.lights=param["shading"]["lights"] if param["shading"] is not None else None
        
        if "cmap" not in param.keys(): #if none or does not exist
            try:
                self.cmap_from_list(param["color_list"])
            except KeyError:
                print("If you do not specify a cmap, you must specify a color_list")
                sys.exit()
        
        self.cmap=matplotlib.cm.get_cmap(param["cmap"]) #cmap object
    
    def set_fractal_parameters(self,param):
        frac_param={}
        frac_param["N"]=param["size"]
        self.frac_size = param["size"]
        frac_param["domain"]=param["domain"]

        frac_param["random"]=param["random"]
        frac_param["func"]=param["func"]
        frac_param["form"]=param["form"]
        frac_param["degree"]=param["degree"]

        frac_param["method"]=param["method"]

        frac_param["tol"]=param["tol"]
        frac_param["damping"]=param["damping"]
        frac_param["itermax"]=param["itermax"]
        frac_param["verbose"]=param["verbose"]

        return frac_param
    
    ### PLOT ###
    def Plot(self,array,name,Dir,print_create=True,**kwargs):
        """Plot a numpy array as an image"""
        fig = plt.figure(frameon=False)
        fig.set_size_inches(1,1)    
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        array = (array * 255/array.max()).astype(np.uint8)
        ax.imshow(array,cmap=self.cmap,**kwargs,vmin=0,vmax=255)

        fig.savefig(Dir+"/"+name,dpi=self.dpi) 
        if print_create==True:
            print("created figure '% s'" %name,end="")
            print("in folder '% s'" %Dir)
        plt.close(fig)
        return None
    
    ### FRACTAL IMAGE ###
    def Fractal_image(self):
        if self.param["verbose"]:
            print("Fractal_image...",end="\r")

        ### Set param
        frac_param=self.set_fractal_parameters(self.param)

        if self.param["raster_image"]!="":
            try:
                orbit_form=np.array(PILIM.open(self.APP_DIR+"/orbit/"+self.param["raster_image"]+".png",).resize((frac_param["N"]+1,frac_param["N"]+1)).convert("L"),dtype=float)
            
            except:
                print("Raster image",self.param["raster_image"],"not found. \nIf you do not want to use a raster image, set 'raster_image' parameters to ''.\n Else check the name of the image in the 'orbit' folder")

        ### General type of Fractal
        if "RFA" in frac_param["method"]: #Root finding algorithm
            frac_obj=RFA_fractal(frac_param)
            frac_param["func"]=frac_obj.coefs
            self.func=frac_obj.coefs

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

                    self.z,conv,dist,normal=frac_obj.Newton_method(d2roots, #z0
                                                            lambda z: np.array([frac_obj.poly.poly(z[i],c_coefs[i]) for i in range(len(c_coefs))]), #f
                                                            lambda z: np.array([frac_obj.poly.dpoly(z[i],c_coefs[i]) for i in range(len(c_coefs))]), #f'
                                                            tol=1.e-05,
                                                            max_steps=50,
                                                            damping=complex(1,0.2),
                                                            verbose = frac_param["verbose"],)


                    self.z,conv,dist=self.z.reshape(shape),conv.reshape(shape),dist.reshape(shape)

                    self.z,conv,dist=self.z.real,conv.real,dist.real

                    #Binary map
                    frac_entire=binary_dilation(canny(conv),iterations=2)    

            ## No subtype specified
            else:
                if "Newton" in frac_param["method"]: #Newton method
                    self.z,conv,dist,normal=frac_obj.Newton_method(frac_obj.array,
                                                        lambda z: frac_obj.poly.poly(z,frac_obj.coefs),
                                                        lambda z: frac_obj.poly.dpoly(z,frac_obj.coefs),
                                                        lambda z: frac_obj.poly.d2poly(z,frac_obj.coefs),
                                                        frac_param["tol"],
                                                        frac_param["itermax"],
                                                        frac_param["damping"],
                                                        orbit_form=orbit_form,
                                                        verbose = frac_param["verbose"],)

                
                elif "Halley" in frac_param["method"]:
                    self.z,conv,dist,normal=frac_obj.Halley_method(frac_obj.array,
                                                        lambda z: frac_obj.poly.poly(z,frac_obj.coefs),
                                                        lambda z: frac_obj.poly.dpoly(z,frac_obj.coefs),
                                                        lambda z: frac_obj.poly.d2poly(z,frac_obj.coefs),
                                                        frac_param["tol"],
                                                        frac_param["itermax"],
                                                        frac_param["damping"],
                                                        orbit_form=orbit_form,
                                                        verbose = frac_param["verbose"],)
            #throw away imaginary part
            self.z,conv=self.z.real,conv.real

            # Julia fractal
        elif "Julia" in frac_param["method"]:
                pass
                
            #Mandelbrot fractal
        elif "Mandelbrot" in frac_param["method"]:
                pass

        #Edge detection
        self.frac_boundary=(canny(conv)+sobel(conv)*(-1) + canny(conv*(-1))+sobel(conv)) # +1e-02 to avoid division by 0
        self.frac_boundary = np.where(self.frac_boundary>0,1,0)

        #Shading
        if self.param["shading"] == "blinn_phong":
            self.shade=self.blinn_phong(normal,self.lights)
        elif self.param["shading"] == "matplotlib":
            self.shade=self.matplotlib_light_source(self.z,self.lights)
        elif self.param["shading"] == "fossil":
            self.shade=self.matplotlib_light_source(self.z*self.frac_boundary,self.lights)

        #Plot
        if self.param["test"]:
            self.Plot(self.shade,self.file_name+"_shader",self.param["dir"],print_create=self.param["verbose"])
            self.Plot(self.frac_boundary,self.file_name+"_nobckg",self.param["dir"],print_create=self.param["verbose"])
            self.Plot(self.frac_boundary*normal,self.file_name+"_shader_nobckg",self.param["dir"],print_create=self.param["verbose"])
            self.Plot(self.z,self.file_name+"_iter",self.param["dir"],print_create=self.param["verbose"])
            


        if self.param["verbose"]:
            print("Fractal_image...Done")
        return self.z

    ############RENDERING#################
    ### COLORS ###
    def cmap_from_list(self,color_list, cmap_name="cmap"):
        """ Create a colormap from a list of colors

        Args:
            colors (list): list of colors RGBA or RGB

        Returns:
            matplotlib.colors.ListedColormap: colormap
        """
        cmap=colors.LinearSegmentedColormap.from_list(cmap_name,color_list)
        cm.register_cmap(name=cmap, cmap=cmap)

        return cmap
    
    def apply_colormap_with_alpha(arr, cmap_name):
        """
        Apply a colormap to a 2D array with an alpha channel.

        Args:
            arr (np.ndarray): 2D array of values
            cmap_name (str): name of a colormap

        Returns:
            np.ndarray: 3D array of RGBA values
        """
        normed_data = (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) # Normalize the array
        mapped_colors = plt.cm.get_cmap(cmap_name)(normed_data) # Apply the colormap by name
        mapped_data = (mapped_colors * 255).astype(np.uint8) # Convert to RGBA and scale to 0-255
        return mapped_data

     ### SHADERS ###
    def matplotlib_light_source(self,array,light=(315,20,1.5,1.2),**kwargs):
        """ Create a matplotlib light source
p
        Args:
            light (tuple): light parameters
            (azimuth, elevation, vert_exag, fraction)

        Returns:
            matplotlib.colors.LightSource: light source
        """

        lightS = colors.LightSource(azdeg=light[0], altdeg=light[1], **kwargs)
        array = lightS.shade(array, cmap=self.cmap, vert_exag=light[2],fraction=light[3],**kwargs)
        return array
    
    def blinn_phong(self,normal, light):
        """ Blinn-Phong shading algorithm
    
        Brightess computed by Blinn-Phong shading algorithm, for one pixel,
        given the normal and the light vectors

        Args:
        normal: complex number
        light: (float, float, float)
                light vector: angle azimuth [0-360], angle elevation [0-90],
                opacity [0,1], k_ambiant, k_diffuse, k_spectral, shininess
           
        Returns:
            float: Blinn-Phong brightness

        from https://github.com/jlesuffleur/gpu_mandelbrot/blob/master/mandelbrot.py
        """
        ## Lambert normal shading (diffuse light)
        normal=np.divide(normal, abs(normal), out=np.zeros_like(normal), where=normal!=0)    
        
        # theta: light azimuth; phi: light elevation
        # light vector: [cos(theta)cos(phi), sin(theta)cos(phi), sin(phi)]
        # normal vector: [normal.real, normal.imag, 1]
        # Diffuse light = dot product(light, normal)
        ldiff = (normal.real*np.cos(light[0])*np.cos(light[1]) +
                normal.imag*np.sin(light[0])*np.cos(light[1]) +
                1*np.sin(light[1]))
        # Normalization
        ldiff = ldiff/(1+1*np.sin(light[1]))
        
        ## Specular light: Blinn Phong shading
        # Phi half: average between pi/2 and phi (viewer elevation)
        # Specular light = dot product(phi_half, normal)
        phi_half = (np.pi/2 + light[1])/2
        lspec = (normal.real*np.cos(light[0])*np.sin(phi_half) +
                normal.imag*np.sin(light[0])*np.sin(phi_half) +
                1*np.cos(phi_half))
        # Normalization
        lspec = lspec/(1+1*np.cos(phi_half))
        #spec_angle = max(0, spec_angle)
        lspec = lspec ** light[6] # shininess
        
        ## Brightness = ambiant + diffuse + specular
        bright = light[3] + light[4]*ldiff + light[5]*lspec
        ## Add intensity
        bright = bright * light[2] + (1-light[2])/2 
        return bright
    
    ### FILTERS ###
    def high_pass_gaussian(self,array,sigma=3):
        """High pass gaussian filter"""
        lowpass = gaussian_filter(array, sigma)
        return array - lowpass
    
    def local_treshold(self,array):
        """Local treshold filter"""
        return array>threshold_local(array)
    
    ### BLENDING ###

    ### POST-IMAGE HANDLER ###
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



if __name__=='__main__':
    import time
    cmap_dict = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                  'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                  'spring', 'summer', 'autumn', 'winter', 'cool','Wistia',
                  'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu']
    
    raster_image_list=["circle","circle2","fire","human","eyes","planet","stars"]
    start_time=time.time()

    i=0
    for k in range(1):
        new_time=time.time()
        print("-----------------",i,"------------------")

        param={
        #### General parameters
        "clean_dir":False, #clean image dir before rendering
        "verbose":False, #print stuff
        "test":True, #for trying stuff and knowing where to put it
        "media_form":"image", #image or video

        #### Image parameters
        #General
        "dir": "images",
        "file_name": f"test{i}",
        "raster_image":np.random.choice(raster_image_list), # if None, raster image is np.zeros((size,size))
        "dpi":1000,

        #Colors
        "color_list":["black","darkgrey","orange","darkred"],
        "cmap":np.random.choice(cmap_dict), #for testing only

        #Shading
        "shading":True,
        "lights": (45., 0, 40., 0, 0.5, 1.2, 1),
        #Filters

        #### Fractal parameters
        "method": "RFA Newton",
        "size": 500,
        "domain":np.array([[-1,1],[-1,1]]),
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
        "damping":complex(1.01,-.01),

        ##Julia parameters
    }
        try:
            obj=IMAGE(param)
            obj.Fractal_image(param)
            print("raster image: ",param["raster_image"])


            # Uncomment to plot 10 images with different cmap
            #for j in range(10):
            #    obj.cmap=np.random.choice(cmap_dict)
            #    print(f"cmap{j}: ",obj.cmap)
            #    obj.Plot(obj.shaded,param["image_name"]+f"_{j}",param["dir"],print_create=False)

        except KeyboardInterrupt:
            sys.exit()
        #print("\nseconds: " ,(time.time() - new_time))
        i+=1
    print("----------------- {}.{:02d} min -----------------".format(int((time.time() - start_time)//60), int((time.time() - start_time)%60)))
