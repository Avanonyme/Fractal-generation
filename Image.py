import os
import sys

import numpy as np
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image as PILIM, ImageDraw

from scipy.ndimage import gaussian_filter,sobel,binary_dilation
from skimage.filters import threshold_local
from skimage.feature import canny

import matplotlib.colors as mcolors
import extcolors as extc
from colorspacious import cspace_convert, cspace_converter
from scipy.spatial import cKDTree
import colorsys
import seaborn as sns

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
    ### SET PARAM###
    def __init__(self,param) -> None:
        if param["verbose"]:
            print("Init Image class...",end="\r")
        ### Create folders
        self.APP_DIR = os.path.dirname(os.path.abspath(__file__))
        self.IM_DIR=os.path.join(self.APP_DIR,param["temp_dir"])

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
                self.cmap=self.cmap_from_list(param["color_list"])
            except KeyError:
                print("If you do not specify a cmap, you must specify a color_list")
                sys.exit()
        else:
            self.cmap = param["cmap"]
        
        #self.cmap=matplotlib.cm.get_cmap(param["cmap"]) #cmap object
    
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
        array = array.real
        fig = plt.figure(frameon=False)
        fig.set_size_inches(1,1)    
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        array = ((array - np.min(array)) / (np.max(array) - np.min(array)) * 255).astype(np.uint8)
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
        if self.param["shading"]["type"] == "blinn-phong":
            self.shade=self.blinn_phong(normal,self.lights)
        elif self.param["shading"]["type"] == "matplotlib":
            self.shade=self.matplotlib_light_source(self.z,self.lights)
        elif self.param["shading"]["type"] == "fossil":
            self.shade=self.matplotlib_light_source(self.z*self.frac_boundary,self.lights)
        self.normal = normal

        #Plot
        if self.param["test"]:
            self.Plot(self.shade,self.file_name+"_shader",self.param["temp_dir"],print_create=self.param["verbose"])
            self.Plot(self.frac_boundary,self.file_name+"_nobckg",self.param["temp_dir"],print_create=self.param["verbose"])
            self.Plot(self.frac_boundary*normal,self.file_name+"_shader_nobckg",self.param["temp_dir"],print_create=self.param["verbose"])
            self.Plot(self.z,self.file_name+"_iter",self.param["temp_dir"],print_create=self.param["verbose"])
            


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
        cmap=mcolors.LinearSegmentedColormap.from_list(cmap_name,color_list)
        cm.register_cmap(name=cmap_name, cmap=cmap)

        return cmap_name
    
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
        blend_mode = kwargs.pop('blend_mode', 'soft')
        norm = kwargs.pop('norm', mcolors.PowerNorm(0.3))

        lightS = mcolors.LightSource(azdeg=light[0], altdeg=light[1], **kwargs)

        #assuming self.cmap is a str
        array = lightS.shade(array, cmap=cm.get_cmap(self.cmap), vert_exag=light[2],fraction=light[3],blend_mode=blend_mode, norm=norm, **kwargs)
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


class COLOUR():
    """
    Generate list of colours hex codes from a list of colours, an image or colour palettes

    """

    def __init__(self) -> None:
        
        pass

    ### PALETTE PROPERTIES ###
    def set_parameters(self,param):
        pass

    
    ### COLOUR TRANSFORMATIONS ###
    def rgb_to_hex(self,rgb):
        hex_color = "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])
        return hex_color
    
    def hex_to_rgb(self,hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (r, g, b)
    
    def rgb_to_hsv(self,r, g, b):
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val

        # Hue calculation
        if delta == 0:
            h = 0
        elif max_val == r:
            h = ((g - b) / delta) % 6
        elif max_val == g:
            h = (b - r) / delta + 2
        else:
            h = (r - g) / delta + 4
        h /= 6

        # Saturation calculation
        if max_val == 0:
            s = 0
        else:
            s = delta / max_val

        # Value calculation
        v = max_val

        return h, s, v
    def hsv_to_rgb(self,h, s, v):
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i %= 6

        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        elif i == 5:
            r, g, b = v, p, q

        return r, g, b

    def lerp_color(self,color1, color2, t):
        r = int((1 - t) * color1[0] + t * color2[0])
        g = int((1 - t) * color1[1] + t * color2[1])
        b = int((1 - t) * color1[2] + t * color2[2])
        return (r, g, b)

    def complementary_color(self,color):
        """
        
        """

        if type(color) == str: #if hex
            hex_color = color
            rgb_color = self.hex_to_rgb(hex_color)
            comp_rgb = tuple(255 - val for val in rgb_color)
            return self.rgb_to_hex(comp_rgb)
        elif type(color) == tuple: #if rgb
            rgb_color = color
            comp_rgb = tuple(255 - val for val in rgb_color)
            return comp_rgb
    
    ### LIST HANDLER ###
    def blend_colors(self,color1, color2, alpha):
        return (
            int((1 - alpha) * color1[0] + alpha * color2[0]),
            int((1 - alpha) * color1[1] + alpha * color2[1]),
            int((1 - alpha) * color1[2] + alpha * color2[2])
        )

    ### PALETTE GENERATION ###
    def create_uniform_colormap(self,colors):
        # Convert the colors from hex to RGB format in the range 0 to 1
        rgb_colors = [self.hex_to_rgb(color) for color in colors]

        # Convert the colors to CIELab space
        colors_CIELab = [cspace_converter("sRGB1", "CIELab")(color) for color in rgb_colors]

        # Create a sequence in CIELab space to represent uniform color progression
        uniform_sequence_CIELab = np.linspace(colors_CIELab[0], colors_CIELab[-1], len(colors))

        # Use a KDTree to find the closest matching color from the original colors to the uniform sequence
        tree = cKDTree(colors_CIELab)
        uniform_colors_index = tree.query(uniform_sequence_CIELab, k=1)[1]

        # Reorder the original colors based on the indices found
        uniform_colors = [colors[i] for i in uniform_colors_index]

        return uniform_colors
    def create_perceptually_uniform_palette(self,colors, steps=256):
        # Convert the colors from hex to RGB format in the range 0 to 1
        rgb_colors = [self.hex_to_rgb(color) for color in colors]

        # Convert the RGB colors to CIELab using the predefined conversion path
        colors_cielab = [cspace_convert(color, start="sRGB255", end="CIELab") for color in rgb_colors]
        # Create a tree from the converted colors
        tree = cKDTree(colors_cielab)
        
        # Interpolate between the colors in CIELab space
        interpolated_colors_cielab = []
        for i in range(steps):
            t = i / (steps - 1)
            indices = tree.query([colors_cielab[0] * (1 - t) + colors_cielab[-1] * t], k=2)[1][0]
            weights = 1 - tree.query([colors_cielab[0] * (1 - t) + colors_cielab[-1] * t], k=2)[0][0]
            color = colors_cielab[indices[0]] * weights[0] + colors_cielab[indices[1]] * (1 - weights[1])
            interpolated_colors_cielab.append(color)

        # Convert the interpolated colors back to sRGB space and then to hex format
        hex_colors = [cspace_convert(color, start="CIELab", end="sRGB255").tolist() for color in interpolated_colors_cielab]
        hex_colors = ['#%02x%02x%02x' % (int(color[0]*255), int(color[1]*255), int(color[2]*255)) for color in hex_colors]

        return hex_colors

    def create_blended_palette(self,original_palette, accent_method):
        blended_palette = []

        if type(original_palette[0]) == str: #if hex
            original_palette = [self.hex_to_rgb(color) for color in original_palette]
        else: #if rgb or rgba
            original_palette = original_palette
        
        if type(accent_method[0]) == str: #if hex
            accent_method = [self.hex_to_rgb(color) for color in accent_method]
        else: #if rgb or rgba
            accent_method = accent_method

        for i in range(len(original_palette)):
            original_color = original_palette[i]
            accent_color = accent_method[i % len(accent_method)]  # Repeating accents if needed

            alpha = i/len(original_palette)
            blended_color = self.blend_colors(original_color, accent_color, alpha)
            blended_palette.append(self.rgb_to_hex(blended_color))

        return blended_palette

    def create_palette_from_colours(self, rgb_colors, num_steps):
        """
        Generate a palette from a short list of colors
        """
        num_colors = len(rgb_colors)

        # define number of steps between each color from num_steps and num_colors
        inter_num_steps = int(num_steps / (num_colors - 1))
        step_size = 1/inter_num_steps

        palette = [rgb_colors[0]]

        for i in range(1, num_colors - 1):
            for j in range(inter_num_steps):
                t = j * step_size
                interpolated_color = self.lerp_color(rgb_colors[i], rgb_colors[i + 1], t)
                palette.append(interpolated_color)

        palette.append(rgb_colors[-1])
        return palette

    def create_palette_from_image(self,image):
        """
        Generate a palette from an image (path, array or PIL image)

        from https://kylermintah.medium.com/coding-a-color-palette-generator-in-python-inspired-by-procreate-5x-b10df37834ae
        """
        def fetch_image(image_path):
            try:
                img = PILIM.open(image_path)
                return img
            except:
                print("Image not found")
                sys.exit()

        # Extract image 
        if type(image) == str: #if image is a path
            img = fetch_image(image)
        else: #if image is an image
            if type(image) == PILIM.Image: #if PIL image
                img = image
            else:
                img = PILIM.fromarray(image)
                #img = image
        #output: PIL image
        
        def extract_colors(img):
            colors, pixel_count = extc.extract_from_image(img)
            return colors
        colors = extract_colors(img)
        #output: array of tuples (RGB, count)

        # format array of tuples to list of RGB
        colors = [color[0] for color in colors]
        num_steps = len(colors)+10

        self.image_palette = self.create_palette_from_colours(colors, num_steps)

        # Create palette
        #colors = self.create_perceptually_uniform_palette(colors)

        palette = self.create_palette_from_colours(colors, num_steps)
        #rgb_to_hex
        palette = [self.rgb_to_hex(color) for color in palette]

        return palette

    def create_complementary_palette(self,hex_colors):
        """
        Generate a complementary palette from a palette
        """
        complementary_palette = []
    
        for hex_color in hex_colors:
            comp_color = self.complementary_color(hex_color)
            complementary_palette.append(comp_color)
    
        return complementary_palette

    def create_triadic_palette(self,colors):
        triadic_color_1 = []
        triadic_color_2 = []
        for color in colors:
            # Convert the selected color to RGB if it's in HEX
            if isinstance(color, str):
                color = self.hex_to_rgb(color)

            # Convert the RGB color to HSV
            h, s, v = self.rgb_to_hsv(*color)

            # Generate two additional colors by adding 120 degrees in the hue component
            triadic_colors = [self.hsv_to_rgb((h + offset) % 1, s, v) for offset in [0, 1/3, 2/3]]
            
            #convert to int
            triadic_colors = [tuple(int(val) for val in color) for color in triadic_colors]

            # Convert the triadic colors to hex
            triadic_color_1.append(self.rgb_to_hex(triadic_colors[1]))
            triadic_color_2.append(self.rgb_to_hex(triadic_colors[2]))

        return triadic_color_1, triadic_color_2
    
    def create_analogous_palette(self,colors, hue_shift=30):
        analogous_palette = []

        for color in colors:
            # Check if the color is given in hex format and convert to RGB
            if isinstance(color, str) and color.startswith("#"):
                color = self.hex_to_rgb(color)

            # Normalize RGB to [0, 1] range
            color = [x/255 for x in color]

            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(*color)

            # Shift the hue by the specified angle (in degrees)
            h = (h + hue_shift/360) % 1

            # Convert back to RGB
            analogous_color = colorsys.hsv_to_rgb(h, s, v)

            # Denormalize RGB to [0, 255] range
            analogous_color = [int(x*255) for x in analogous_color]

            analogous_palette.append(self.rgb_to_hex(analogous_color))

        return analogous_palette

    def create_split_complementary_palette(self,colors, hue_shift=30):
        split_complementary_palette_1 = []
        split_complementary_palette_2 = []

        for color in colors:
            # Check if the color is given in hex format and convert to RGB
            if isinstance(color, str) and color.startswith("#"):
                color = self.hex_to_rgb(color)

            # Normalize RGB to [0, 1] range
            color = [x/255 for x in color]

            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(*color)

            # Find the complementary hue
            complementary_hue = (h + 0.5) % 1

            # Find the split complementary hues by shifting the complementary hue
            split_hue1 = (complementary_hue + hue_shift/360) % 1
            split_hue2 = (complementary_hue - hue_shift/360) % 1

            # Convert back to RGB
            split_color1 = [int(x*255) for x in colorsys.hsv_to_rgb(split_hue1, s, v)]
            split_color2 = [int(x*255) for x in colorsys.hsv_to_rgb(split_hue2, s, v)]

            split_complementary_palette_1.append(self.rgb_to_hex(split_color1))
            split_complementary_palette_2.append(self.rgb_to_hex(split_color2))

        return split_complementary_palette_1, split_complementary_palette_2

    def create_tetradic_palette(self,colors):
        tetradic_palette = []

        for color in colors:
            # Check if the color is given in hex format and convert to RGB
            if isinstance(color, str) and color.startswith("#"):
                color = self.hex_to_rgb(color)

            # Normalize RGB to [0, 1] range
            color = [x/255 for x in color]

            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(*color)

            # Create the tetradic hues
            tetradic_hues = [(h + i/4) % 1 for i in range(4)]

            # Convert back to RGB
            tetradic_colors = [colorsys.hsv_to_rgb(hue, s, v) for hue in tetradic_hues]

            # Denormalize RGB to [0, 255] range, convert to hex
            tetradic_colors = [self.rgb_to_hex(tuple(int(x*255) for x in color)) for color in tetradic_colors]


            tetradic_palette.append(tetradic_colors)


        #[[1,2,3,4],[1,2,3,4]] -> [[1,1],[2,2],[3,3],[4,4]]
        tetradic_palette = list(zip(*tetradic_palette))
        return tetradic_palette[0], tetradic_palette[1], tetradic_palette[2], tetradic_palette[3]

    def create_shades_palette(self,colors, steps=5):
        shades_palette = []

        for color in colors:
            # Check if the color is given in hex format and convert to RGB
            if isinstance(color, str) and color.startswith("#"):
                color = self.hex_to_rgb(color)

            # Normalize RGB to [0, 1] range
            color = [x/255 for x in color]

            # Convert RGB to HSV
            h, s, v = colorsys.rgb_to_hsv(*color)

            # Create shades by varying the value
            shades_colors = [colorsys.hsv_to_rgb(h, s, v * (i/steps)) for i in range(steps+1)]

            # Denormalize RGB to [0, 255] range
            shades_colors = [self.rgb_to_hex(tuple(int(x*255) for x in color)) for color in shades_colors]

            shades_palette+=shades_colors

        return shades_palette
   
    def create_accents_palette(self,palette, accent_method):
        '''
        Generate an accents palette by blending method and gives uniformity to the palette

        Args:
            palette (list): list of colors hex
            accent_method (str): combination of "complementary", "triadic", "analogous", "split_complementary", "tetradic", "shades"
        '''
        
        if "complementary" in accent_method:
            palette += self.create_blended_palette(palette, self.create_complementary_palette(palette))
        if "triadic" in accent_method:
            palette += self.create_blended_palette(palette, self.create_triadic_palette(palette)[0])
            palette += self.create_blended_palette(palette, self.create_triadic_palette(palette)[1])
        if "analogous" in accent_method:
            palette += self.create_blended_palette(palette, self.create_analogous_palette(palette))
        if "split_complementary" in accent_method:
            palette += self.create_blended_palette(palette, self.create_split_complementary_palette(palette)[0])
            palette += self.create_blended_palette(palette, self.create_split_complementary_palette(palette)[1])
        if "tetradic" in accent_method:
            palette += self.create_blended_palette(palette, self.create_tetradic_palette(palette)[0])
            palette += self.create_blended_palette(palette, self.create_tetradic_palette(palette)[1])
            palette += self.create_blended_palette(palette, self.create_tetradic_palette(palette)[2])
            palette += self.create_blended_palette(palette, self.create_tetradic_palette(palette)[3])
        if "shades" in accent_method:
            palette += self.create_blended_palette(palette, self.create_shades_palette(palette))

        return palette
 
    def get_seaborn_cmap(self,palette_or_colors):

        if isinstance(palette_or_colors, str):
            # If a string (palette name) is provided, directly use it to get the Seaborn colormap
            cmap = sns.color_palette(palette_or_colors).as_hex()
        else:
            # If a list of colors is provided, we'll use it to create a colormap
            if all(isinstance(color, str) and color.startswith("#") for color in palette_or_colors):
                hex_colors = palette_or_colors
            else:
                # If colors are provided in RGB, convert them to hex
                hex_colors = [mcolors.to_hex(color) for color in palette_or_colors]
            
            # Create a colormap using Seaborn with the provided hex colors
            cmap = sns.color_palette(hex_colors).as_hex()

        return cmap
    
    def get_matplotlib_cmap(self,palette_or_colors):
        if isinstance(palette_or_colors, str):
            # If a string (palette name) is provided, use it to get the Matplotlib colormap
            cmap = plt.get_cmap(palette_or_colors)
            hex_colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]
        else:
            # If a list of colors is provided, convert them to hex if needed
            hex_colors = [mcolors.to_hex(color) if isinstance(color, tuple) else color for color in palette_or_colors]

        return hex_colors
    
    ### RENDER ###
    def cmap_from_list(self,color_list, cmap_name="my_cmap"):
        """ Create a matplotlib colormap objectfrom a list of colors

        Args:
            colors (list): list of colors hex

        Returns:
            matplotlib.colors.ListedColormap: colormap
        """
        if type(color_list[0]) == str: #if hex
            color_list = [self.hex_to_rgb(color) for color in color_list]
            color_list = [(r/255, g/255, b/255) for r, g, b in color_list]
        else: #rgb or rgba
            pass
        cmap=mcolors.LinearSegmentedColormap.from_list(cmap_name,color_list,N=256)
        cm.register_cmap(name=cmap_name, cmap=cmap)


        return cmap_name
    
    def render_color_palette(self,hex_colors, palette_name = "palette"):
        num_colors = len(hex_colors)
        color_map = []
        
        for hex_color in hex_colors:
            color_map.append(self.hex_to_rgb(hex_color) if type(hex_color) == str else hex_color)
        fig, ax = plt.subplots(figsize=(num_colors * 0.5, 1))  # Adjust the figure size
        
        # Create a color bar with the specified hex colors
        cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=mcolors.ListedColormap(color_map)),
                        cax=ax, orientation='horizontal', ticks=[])
        cb.outline.set_visible(False)
        
        plt.savefig("images/"+palette_name + ".png", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

        return PILIM.open("images/"+palette_name + ".png")

    def stack_palettes(self,palette1, palette2, multiple_palette1 = False):
        # Create an instance of your class (replace 'YourClassName' with the actual name)
        colour_obj = COLOUR()
        
        # Render the two palettes using the existing function
        if multiple_palette1: # we want to stack multiple palettes
            palette1_image = PILIM.open("images/palettes.png")
        
        else:
            palette1_image = colour_obj.render_color_palette(palette1, palette_name="palette1")
        palette2_image = colour_obj.render_color_palette(palette2, palette_name="palette2")
        
        # Stack the two images vertically
        stacked_image = PILIM.new('RGB', (palette1_image.width, palette1_image.height + palette2_image.height))
        stacked_image.paste(palette1_image, (0, 0))
        stacked_image.paste(palette2_image, (0, palette1_image.height))

        # Save the image
        stacked_image.save("images/palettes.png")

        #delete images
        if not multiple_palette1:
            os.remove("images/palette1.png")
        os.remove("images/palette2.png")
        
        return PILIM.open("images/palettes.png")

    def get_colors_from_cmap(self,cmap_name, n_colors=256):
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i) for i in range(n_colors)]
        return colors


def COLOUR_wrapper(palette_name,method = "accents",simple_cmap = False,**kwargs):
    """
    Wrapper for the COLOUR.create_palette_from_image func. Depending on the method args
    more function can be called

    Args:
        palette_name (str): path to image or name of the palette
        method (str): method(s) name (seaborn, matplotlib, accents,)
        **kwargs: method args
    """
    accent_method = kwargs.pop('accent_method', "complementary")
    simple_cmap = kwargs.pop('simple_cmap', False)

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

