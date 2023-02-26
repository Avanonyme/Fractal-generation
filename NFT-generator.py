import os
import json
import sys

import random
import numpy as np

from  PIL import Image as PILIM
from Im_logic import Video_maker,Image,paste_image

parameters={
    "Repetition":1,
    "FPS":30 ,
    "Duration":5, #seconds
    "zoom":1.2,
    "color_API_tuple":[[64,15,88],(15,5,0),(249,44,6),"N","N"],  #Example: [[r,g,b],"N","N","N","N"]
    "cmap":None,
    "coord":np.array([[-1,1],[-1,1]]),
    "degree": random.randint(5,20),
    "rand_coef": True,
    "coef": None, #Must have value if rand_coef==False
    "dpi": 500,
    "itermax":150,

}

# Opening JSON file
finished_dir=os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)),"Finished/Finished/")
metadata_folder=os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)),"Finished/metadata/")
img_folder=os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)),"Finished/img/")
try:
    for i in range(18,21):

        with open(metadata_folder+f'{i}.json') as json_file:
            data = json.load(json_file)["attributes"]
            parameters["attribute"]=data[0]["value"]
        if data[0]["value"]=="Video":
            file_dir=Video_maker(parameters)
        elif data[0]["value"]=="Image":
            file_dir,z=Image(parameters)

        f=0
        from natsort import natsorted
        listdir=natsorted(os.listdir(file_dir))
        for files in listdir:
            try:os.mkdir(finished_dir+f"{i}/")
            except:pass

            foreground=PILIM.open(img_folder+f"{i}.png")
            bckg=PILIM.open(file_dir+files)
            im=paste_image(bckg,foreground,0,0,bckg.size[0],bckg.size[1])
            im.save(finished_dir+f"{i}/"+f"frame{f}.png") #MUST NAME FRAME
            f+=1
except KeyboardInterrupt:
        sys.exit()

