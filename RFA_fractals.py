
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import random as rd
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_mean, butterworth
import itertools as it

#plot handling
def plot(array,name,cmap=LinearSegmentedColormap.from_list("lambdacmap",["black","white"],N=2),dpi=1000):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1,1)    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(array,cmap=cmap)
    fig.savefig(f"{name}.png",dpi=dpi)

class Polynomials():
    def __init__(self,func=[],random_poly=True,form="root",**kwargs):
        """
        func: list of roots or coefficients. If form=="taylor_approx", func is a function (ex. lambda z: sin(z))
        random_poly: True or False (if True, func is ignored)
        form: "root","coef","taylor_approx" (if random_poly==True, form is ignored)
        """
        self.__dict__.update(kwargs)


        if random_poly==True:
            self.coefs=self.choose_poly(self.__dict__.get("max_degree"))
            self.roots=self.coefficients_to_roots(self.coefs)
        
        else: #random_poly==False
            if func==[]:
                raise ValueError("No function given")

            if form=="root":
                self.roots=func
                self.coefs=self.roots_to_coefficients(self.roots)                

            elif form=="taylor_approx":
                self.coefs=self.taylor_approximation(func,degree=self.__dict__.get("degree"),interval=self.__dict__.get("interval"))
                self.roots=self.coefficients_to_roots(self.coefs)
            else: #form=="coef" or form==None
                #coefs by default
                self.coefs=func
                self.roots=self.coefficients_to_roots(self.coefs)        
        

        ### POLYNOMIALS ###
    # Code taken from https://github.com/3b1b/videos/blob/master/_2022/quintic/roots_and_coefs.py

    def roots_to_coefficients(self,roots):
        """Convert roots form poly to coefficients form poly"""
        n = len(list(roots))
        return [
            ((-1)**(n - k)) * sum(
                np.prod(tup)
                for tup in it.combinations(roots, n - k)
            )
            for k in range(n)
        ] + [1]

    def poly(self,z, coefs):
        """Evaluate polynomial at z with coefficients coefs"""
        return sum(coefs[k] * z**k for k in range(len(coefs)))


    def dpoly(self,z, coefs):
        """Derivative of poly(z, coefs)"""
        return sum(k * coefs[k] * z**(k - 1) for k in range(1, len(coefs)))

    def d2poly(self,z,coefs):
        """Second derivative of poly(z, coefs)"""
        return sum(k*(k-1)*coefs[k]*z**(k-2) for k in range(2,len(coefs)))

    def find_root(self,func, dfunc, seed=complex(1, 1), tol=1e-8, max_steps=100):
        # Use newton's method
        last_seed = np.inf
        for n in range(max_steps):
            if abs(seed - last_seed) < tol:
                break
            last_seed = seed
            seed = seed - func(seed) / dfunc(seed)
        return seed

    def coefficients_to_roots(self,coefs):
        """Find roots of polynomial with coefficients coefs"""
        if len(coefs) == 0:
            return []
        elif coefs[-1] == 0:
            return self.coefficients_to_roots(coefs[:-1])
        roots = []
        # Find a root, divide out by (z - root), repeat
        for i in range(len(coefs) - 1):
            root = self.find_root(
                lambda z: self.poly(z, coefs),
                lambda z: self.dpoly(z, coefs),
            )
            roots.append(root)
            new_reversed_coefs, rem = np.polydiv(coefs[::-1], [1, -root])
            coefs = new_reversed_coefs[::-1]
        return roots

    def taylor_approximation(self,func,degree,interval):
        """Taylor approximation of function func"""
        if degree is None:
            degree=10
        if interval is None:
            interval=1
        
        from scipy.interpolate import approximate_taylor_polynomial

        return approximate_taylor_polynomial(func,0,degree,interval).c

    def choose_poly(self,max_degree):
        """Choose random polynomial"""
        if max_degree is None:
            max_degree=8
        elif max_degree<4:
            print("max_degree must be >=4. Setting to 8")
            max_degree=8
        
        degree=rd.randint(4,max_degree)
        coefs=[complex(rd.uniform(-10,10),rd.uniform(-10,10)) for i in range(degree)]
        return coefs

    def add_c_to_coefs(self,c,func,random=True,c_expression=None):
        """Add c to coefficients of polynomial
            if random is False, c_expression must be list like lambda c: op(coefs,expression of c)
        
        """
        print("Adding c to coefficients",end="\r")
        c_coefs=np.zeros((len(c),len(func)),dtype=complex)
        if random==True:
            from operator import add, sub
            ops = (add, sub)
            op = np.random.choice(ops, p=[0.5, 0.5],size=len(func))

            randint=np.random.uniform(0,1,size=len(func))

            mask=np.random.choice([0,1],p=[0.7,0.3],size=len(func)) #mask for 0 or 1
            if np.all(mask==0):
                mask[0]=1;mask[-1]=1
            
            for i,point in enumerate(c):
                    for n in range(len(func)):
                        #debug
                        if i==10 and mask[n]:
                            print(f"coef of z^{n}",np.around(func[n],2),"*",op[n].__name__,"(c,",np.around(randint[n],2),")")
                        elif i==10 and ~mask[n]:
                            print(f"coef of z^{n}",np.around(func[n],2))
                        #end debug

                        c_coefs[i,n]=op[n](point,randint[n])*func[n]*mask[n]+func[n]*(~mask[n]) 

        elif random==False:
            if c_expression is None or c_expression==[]:
                raise ValueError("No c_expression given")
            for i,point in enumerate(c):
                c_coefs[i]=c_expression(point)*np.array(func)
        
        print("Adding c to coefficients...Done")
        return c_coefs

class RFA_fractal():

    def __init__(self, config, **kwargs):
        if config["verbose"]:
            print("Initializing RFA fractal...",end="\r")
        
        self.config = config
        self.__dict__.update(kwargs)

        self.array = self.init_array(config["N"],config["domain"])
        self.poly = Polynomials(func=config["func"],random_poly=config["random"],form=config["form"],**kwargs)
        self.coefs=self.poly.coefs        
        #check if poly converges ok with sample of domain
        if config["random"]==True:
            count=0

            up_treshold=0
            min_treshold=0
            while True:
                if config["verbose"]:
                    print("Choose Polynomial...",end="\r")
                z=self.init_array(100,config["domain"])


                up_treshold=0.6
                min_treshold=0.10
                z,u1,u2,u3=self.Newton_method(z,
                                                lambda z: self.poly.poly(z,self.coefs),
                                                lambda z: self.poly.dpoly(z,self.coefs),
                                                tol=1.e-05,
                                                max_steps=50,
                                                d2func= lambda z: self.poly.d2poly(z,self.coefs),
                                                verbose=False)

                gen_area=z > threshold_mean(z)

                if np.mean(gen_area)>up_treshold:
                    pass
                elif np.mean(gen_area)<min_treshold:
                    pass     
                else:
                    break
                count+=1
                if count>25:
                    break
            if config["verbose"]:
                print("Choose Polynomial...Done",end="\n\n")

                print("Gen_area",np.mean(gen_area))
                print("Chosen roots:",np.around(self.poly.roots,2))
                print("\n")
        else:
            if config["verbose"]:
                print("Chosen roots:",np.around(self.poly.roots,2))
                print("\n")
        if config["verbose"]:
            print("Initializing RFA fractal...Done")

    def init_array(self,N,domain):
        """create array of complex numbers"""
        real_dom=np.linspace(domain[0,0],domain[0,1],N,endpoint=False) #expanded domain
        complex_dom=np.linspace(domain[1,0],domain[1,1],N,endpoint=False)
        return np.array([(item+complex_dom*1j) for i,item in enumerate(real_dom)]).reshape(N,N).transpose() #array of shape (N,N)
    
    def get_distance(self ,z, distance_map,option = 4):
        """
        Get distance of all point in array to closest point of some shape (circle, rectangle, etc)
        shape must be in distance space
        """


        width, height = distance_map.shape
        x, y = (width-1)/2*(np.clip(z.real,-1,1)+1), (height-1)/2*(np.clip(z.imag,-1,1)+1)
        i, j = np.int64(x), np.int64(y)
        

        if option==0:
            distance = distance_map[i,j]*z
        
        elif option==1:
            distance = distance_map[i,j] # bitmap
            
        elif option==2:
            distance = distance_map[i,j]*(1-np.abs(x-i))*(1-np.abs(y-j)) #idk weird stuff

        elif option==3:
            distance = distance_map[i,j]*np.arctan(z)

        elif option==4:
            distance=distance_map[i,j]*np.sin(z.real)*np.cos(z.imag)/np.arctan(z) # small smoothing 

        elif option==5:
            distance=distance_map[i,j]*1/np.sqrt(2*np.pi*distance_map[i,j]**2)*np.exp(-z**2/(2*distance_map[i,j]**2)) #gaussian

        elif option==6:
            distance= butterworth(distance_map[i,j]*z) #lowpass filter

        return distance.flatten()

    def Newton_method(self,array,func,dfunc,d2func,tol=1e-08,max_steps=100,damping=1,orbit_form=None,verbose=True):
        """
        Newton method for Nova Fractal
        
        array: array of complex numbers, such that f''(z)=0
        func: poly function to be iterated
        dfunc: derivative of poly function
        tol: scalar,tolerance for convergence
        max_steps: scalar,maximum number of iterations
        damping: complex number,damping factor
        orbit_form: binary mask of shape (N,N).
        verbose: boolean, if True, print progress
        """
        if verbose:
            print("RFA-Newton method...",end="\r")
        #initialisation
        if orbit_form is None:
            orbit_form = np.ones_like(array,dtype=bool)
        shape=array.shape        
        z=array.copy().flatten() #such that f''(z)=0
        prec=np.ones_like(z) #precision
        activepoint=np.ones_like(z)  #used to stop points that attained required precision
        i=0 #count, used to calculate the fractal
        ziter=np.zeros_like(z)
        dz=np.ones_like(z)


        dist=1e20*np.ones_like(z)
        
        distance_map = distance_transform_edt(np.logical_not(orbit_form))# Compute the distance map
        distance_map=np.divide(distance_map, abs(distance_map), out=np.zeros_like(distance_map), where=distance_map!=0)



        #Newton Method
        while i<=max_steps:

            #checking for points precision reaching tol 
            e=np.where(prec<tol) #checking which points converged
            activepoint[e]=0 #taking out those points 

            ziter[e]=i #noting the count value of those points

            prec[e]=100 #taking out the precision at those points

            activeindex=activepoint.nonzero() #updating the active indexes
            
            #Newton Method
            dx=-damping*func(z)/dfunc(z)
            z[activeindex]=z[activeindex]+dx[activeindex] #if nova add original array

            prec[activeindex]=abs(dx[activeindex])
            
            if verbose:
                print("RFA-Newton method...",i,end="\r")

            #Orbit Trap
            #normal
            dz[activeindex]*= 2-dfunc(dist[activeindex])/d2func(dist[activeindex])
            #method: distance from shape

            dist=np.minimum(dist,self.get_distance(z,distance_map))
            i+=1
            
        #Assigning a value to points that haven't converged
        ziter[activepoint==True]=i #escape method coloring
        z[activepoint==True]=0 #Root coloring
        z=np.around(z,4)
        if verbose:
            print("RFA-Newton method...Done")
        normal=dist/dz
        z,ziter,dist,normal=z.reshape(shape),ziter.reshape(shape),dist.reshape(shape),normal.reshape(shape)
        return ziter,z,np.sqrt(dist),normal


    def Halley_method(self,array,func,dfunc,d2func,tol=1e-08,max_steps=100,damping=1,orbit_form=None,verbose=True):
        """Halley method"""

        if verbose:
            print("RFA-Halley method...",end="\r")
        #initialisation
        if orbit_form is None:
            orbit_form = np.ones_like(array,dtype=bool)
        shape=array.shape        
        z=array.copy().flatten() #such that f''(z)=0
        prec=np.ones_like(z) #precision
        activepoint=np.ones_like(z)  #used to stop points that attained required precision
        i=0 #count, used to calculate the fractal
        ziter=np.zeros_like(z)
        dz=np.ones_like(z)


        dist=1e20*np.ones_like(z)
        
        distance_map = distance_transform_edt(np.logical_not(orbit_form))# Compute the distance map
        distance_map=np.divide(distance_map, abs(distance_map), out=np.zeros_like(distance_map), where=distance_map!=0)



        #Halley Main loop
        while i<=max_steps:

            #checking for points precision reaching tol 
            e=np.where(prec<tol) #checking which points converged
            activepoint[e]=0 #taking out those points 

            ziter[e]=i #noting the count value of those points

            prec[e]=100 #taking out the precision at those points

            activeindex=activepoint.nonzero() #updating the active indexes
            
            #Halley Method
            dx=-damping*2*func(z)*dfunc(z)/(2*dfunc(z)**2-d2func(z)*func(z))
            z[activeindex]=z[activeindex]+dx[activeindex] #if nova add original array

            prec[activeindex]=abs(dx[activeindex])
            
            if verbose:
                print("RFA-Halley method...",i,end="\r")

            #Orbit Trap
            #normal
            dz[activeindex]*= 2-dfunc(dist[activeindex])/d2func(dist[activeindex])
            #dz[activeindex]*=(2*dfunc(dist[activeindex])**2-2*d2func(dist[activeindex])*dfunc(dist[activeindex])+d2func(dist[activeindex])**2)/(2*dfunc(dist[activeindex])**2-d2func(dist[activeindex])*func(dist[activeindex]))**2
            #method: distance from shape
            dist=np.minimum(dist,self.get_distance(z,distance_map))

            i+=1
            
        #Assigning a value to points that haven't converged
        ziter[activepoint==True]=i #escape method coloring
        z[activepoint==True]=0 #Root coloring
        z=np.around(z,4)
        if verbose:
            print("RFA-Halley method...Done")
        normal=dist/dz
        z,ziter,dist,normal=z.reshape(shape),ziter.reshape(shape),dist.reshape(shape),normal.reshape(shape)
        return ziter,z,np.sqrt(dist),normal
        


    #def Secant_method(self,tol=1.e-8,max_steps=100,damping_factor=complex(0.1,0.1),pixel=complex(0.1,0.1),verbose=True)):
