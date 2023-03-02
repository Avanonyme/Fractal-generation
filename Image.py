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
        try: os.mkdir(self.IM_DIR)
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
    
    def Fractal_image(self, parameters):
        print("Fractal_image (IM-Fim)...")

        try: os.mkdir(self.FRAC_DIR)
        except: pass

        frac_param={"degree":parameters["degree"],
                    "coord":parameters["coord"],
                    "dpi":parameters["dpi"],
                    "rand_coef":parameters["rand_coef"],
                    "coef":parameters["coef"]}

        frac_obj=RFA_fractal(frac_param)

        self.z,conv=frac_obj.Newton_method()

        self.z,conv=self.z.real,conv.real
        self.z=abs(self.z-np.max(self.z))

        self.file_name="fractal_array"
        self.Plot(self.z,self.file_name,parameters["dir"])
        self.Plot(conv,"convergence",parameters["dir"])

        print("Done (IM-Fim)")

        return conv

    def Image_maker(self,parameters):
        parameters["dir"]=self.IM_DIR+"/fractal"
        conv=self.Fractal_image(parameters)
        




        # Apply filters
        #conv=self.Unsharp_masking(conv,sigma=1.5,amount=1)
        #self.Anisotropic_diffusion(self.z,niter=10,kappa=70,gamma=0.25,step=(1.,1.),option=1)
        #self.Plot(self.z,"anisotropic_diffusion",parameters["dir"])
        #Binary map
        conv=self.Local_treshold(conv) 
        frac_entire=binary_dilation((canny(conv)+self.Local_treshold(conv*(-1)) +sobel(conv)),iterations=2)
        # Save image
        self.Plot(frac_entire,"canny",parameters["dir"])
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
    def Rendering_3D(self,parameters,image):
        "takes 2D image as input and output 3D heightmap"
        pass
    ### COLORS ###

    ### FILTERS ###
    def Local_treshold(self,array):
        """Local treshold filter"""
        #mask=array>threshold_otsu(self.z) 
        #array=(mask)*array
        return array>threshold_local(array)

    def Contrast(self,array,niter=10,kappa=50,gamma=0.1,step=(1.,1.),option=1):
        """Contrast filter"""
        return array-self.Anisotropic_diffusion(array,niter=niter,kappa=kappa,gamma=gamma,step=step,option=option)
    
    def Anisotropic_diffusion(self,array,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1):
        """
        Anisotropic diffusion.
    
        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)
    
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                        2 Perona Malik diffusion equation No 2
                ploton - if True, the image will be plotted on every iteration
    
        Returns:
                imgout   - diffused image.
    
        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
    
        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)
    
        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes
    
        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.
    
        Reference: 
        P. Perona and J. Malik. 
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 
        12(7):629-639, July 1990.
    
        Original MATLAB code by Peter Kovesi  
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>
    
        Translated to Python and optimised by Alistair Muldal
        Department of Pharmacology
        University of Oxford
        <alistair.muldal@pharm.ox.ac.uk>
    
        June 2000  original version.       
        March 2002 corrected diffusion eqn No 2.
        July 2012 translated to Python
        """
        # initialize output array
        img = array.astype('float32')
        imgout = img.copy()
    
        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()
    
        for ii in range(niter):
    
            # calculate the diffs
            deltaS[:-1,: ] = np.diff(imgout,axis=0)
            deltaE[: ,:-1] = np.diff(imgout,axis=1)
    
            # conduction gradients (only need to compute one per dim!)
            if option == 1:
                gS = np.exp(-(deltaS/kappa)**2.)/step[0]
                gE = np.exp(-(deltaE/kappa)**2.)/step[1]
            elif option == 2:
                gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
                gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
    
            # update matrices
            E = gE*deltaE
            S = gS*deltaS
    
            # subtract a copy that has been shifted 'North/West' by one
            # pixel. don't ask questions. just do it. trust me.
            NS[:] = S
            EW[:] = E
            NS[1:,:] -= S[:-1,:]
            EW[:,1:] -= E[:,:-1]
    
            # update the image
            imgout += gamma*(NS+EW)

    
        return imgout

    def Unsharp_masking(self,array, sigma=1, amount=1):
        """Unsharp masking filter"""
        blurred = gaussian_filter(array, sigma=sigma)
        return array + amount * (array - blurred)

    ### SHADERS ###
    def Phong_shader(self,array,light_direction,base_color):
        from scipy.spatial import Delaunay

        # Calculate the normal vectors at each vertex of the mesh
        tri = Delaunay(array)
        normals = np.cross(tri.points[tri.vertices[:,1]] - tri.points[tri.vertices[:,0]],
                        tri.points[tri.vertices[:,2]] - tri.points[tri.vertices[:,0]])

        # Calculate the illumination at each vertex using Phong shading
        illumination = np.sum(light_direction * normals, axis=1)

        # Apply the illumination to the colors of each vertex
        colors = base_color * illumination
        #Does this works?

    def Fresnel_shader(self,array,light_direction,base_color):
        pass
    def Normal_mapping(self,array,light_direction,base_color):
        pass
        

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
        obj.Image_maker(parameters)

    except KeyboardInterrupt:
        sys.exit()