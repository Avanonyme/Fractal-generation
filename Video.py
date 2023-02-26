import os
import sys

import random
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image as PILIM

from scipy import ndimage
from skimage.filters import threshold_mean
from skimage.feature import canny

from RFA_fractals import RFA_fractal
from Image import IMAGE,Plot


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
    def __init__(self,parameters) -> None:
        print("Init Videos class(V-Init) fractals...")

        ### Create folders
        self.APP_DIR = os.path.dirname(os.path.abspath(__file__))
        self.IM_DIR=os.path.join(self.APP_DIR,"images")
        try: os.mkdir(self.IM_DIR)
        except: pass

        self.FRAME_DIR=os.path.join(self.IM_DIR,"frames")
        try: os.mkdir(self.FRAME_DIR)
        except: pass
        clean_dir(self.FRAME_DIR)

        ### Set paramaters
        self.dpi=parameters["dpi"]
        self.cmap=parameters["cmap"]
        self.fps=parameters["FPS"]
        self.seconds=parameters["Duration"]

        self.frame_name=1
        self.zoom=1
        self.zoom_speed=parameters["zoom"]

        print("Done (V-Init)")

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
    
    def Video_maker(self,parameters):
        print("Video maker (V-vm)...",end="")
        for i in range(parameters["Repetition"]):
            print("V-vm",i+1,"/",parameters["Repetition"])
            # Extract params
            low_obj_param={"degree":parameters["degree"],
                        "coord":parameters["coord"],
                        "dpi":200,
                        "rand_coef":True,
                        "coef":parameters["coef"],
                        }

            #Choose ok polynomial + begin coord
            low_obj=RFA_fractal(low_obj_param)
        
            parameters["coef"]=low_obj.coef
            low_z=low_obj.z

                #Check coords            
            parameters["coord"],parameters["pts"]=self.check_coord(low_z,self.dpi,parameters["coord"],self.zoom)

            #Create video
            parameters["rand_coef"]=False
            parameters["dir"]=self.FRAME_DIR
            for _ in range(self.fps*self.seconds):
                print(_,end="\r")

                #Create frame
                Imobj=IMAGE(parameters) 
                Imobj.Fractal_image(parameters)

                PILIM.open(parameters["dir"]+Imobj.file_name).save(parameters["dir"]+f"frame_{self.frame_name}.png")

                self.zoom=self.zoom/self.zoom_speed
                parameters["coord"],parameters["pts"]=self.check_coord(self.z,self.dpi,parameters["coord"],self.zoom)

                #update frame name
                self.frame_name+=1



            #Copy frame dir to Desktop
            from time import strftime,gmtime
            import shutil
            dt_gmt = strftime("%Y_%m_%d_%H_%M_%S", gmtime())

            src_dir=self.FRAME_DIR
            dest_dir=os.path.join(self.APP_DIR,f"video/video_{dt_gmt}")
            shutil.copytree(src_dir, dest_dir)

            #Create video
            self.Create_video(dest_dir,parameters["FPS"])
        
        print("Done (V-vm)")

    ### ANIMATIONS ###
    def Pulsing(self,parameters,image):
        '''Generate pulsing animation in input image'''
        print('Pulsing (V-P)...',end="")

        

if __name__=='__main__':
    parameters={
    "attribute":"Pulsing",
    "Repetition":1,
    "FPS":30 ,
    "Duration":5, #seconds
    "zoom":1.1,
    "color_API_tuple":[[64,15,88],(15,5,0),(249,44,6),"N","N"],  #Example: [[r,g,b],"N","N","N","N"]
    "cmap":"viridis",
    "coord":np.array([[-1,1],[-1,1]]),
    "degree": random.randint(5,20),
    "rand_coef": True,
    "coef": None, #Must have value if rand_coef==False
    "dpi": 500,
    "itermax":150,
    }

    try:
        obj=Videos(parameters)
        obj.Video_maker(parameters)

    except KeyboardInterrupt:
        sys.exit()
        