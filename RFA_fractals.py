
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import random
import time
from skimage.filters import threshold_mean
from scipy.optimize import root_scalar
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

            elif form=="coefs":
                self.coefs=func
                self.roots=self.coefficients_to_roots(self.coefs)

            elif form=="taylor_approx":
                self.coefs=self.taylor_approximation(funcm,degree=self.__dict__.get("degree"),interval=self.__dict__.get("interval"))
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
            return coefficients_to_roots(coefs[:-1])
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
            degree=5
        if interval is None:
            interval=1
        
        from scipy.interpolate import approximate_taylor_polynomial

        return approximate_taylor_polynomial(func,0,degree,interval).c

    def choose_poly(self,max_degree):
        """Choose random polynomial"""
        if max_degree is None:
            max_degree=8
        elif max_degree<2:
            print("max_degree must be >=2. Setting to 8")
            max_degree=8
        
        degree=random.randint(2,max_degree)
        coefs=[complex(random.uniform(-10,10),random.uniform(-10,10)) for i in range(degree)]
        return coefs

class RFA_fractal():

    def __init__(self, config, **kwargs):
        print("Initializing RFA fractal...",end="\r")
        
        self.config = config
        self.__dict__.update(kwargs)

        self.array = self.init_array(config["N"],config["domain"])

        self.poly = Polynomials(func=config["func"],random_poly=config["random"],form=config["form"],**kwargs)
        self.coefs=self.poly.coefs
        #check if poly converges ok with sample of domain
        if config["random"]==True:
            count=0
            while True:
                print("Choose Polynomial...",end="\r")
                z=self.init_array(100,config["domain"])

                if config["method"]=="Newton":
                    up_treshold=0.35
                    min_treshold=0.12
                    z=self.Nova_Newton_method(z,lambda z: self.poly.poly(z,self.coefs),lambda z: self.poly.dpoly(z,self.coefs),tol=1.e-05,max_steps=50)
                elif config["method"]=="Haley": #SAME FOR NOW
                    up_treshold=0.35
                    min_treshold=0.12
                    z=self.Nova_Halley_method(z,lambda z: self.poly.poly(z,self.coefs),lambda z: self.poly.dpoly(z,self.coefs),tol=1.e-05,max_steps=50)

                elif config["method"]=="Secant": #SAME FOR NOW
                    up_treshold=0.35
                    min_treshold=0.12
                    z=self.Nova_Secant_method(z,lambda z: self.poly.poly(z,self.coefs),lambda z: self.poly.dpoly(z,self.coefs),tol=1.e-05,max_steps=50)


                gen_area=z > threshold_mean(z)

                if np.mean(gen_area)>up_treshold:
                    break
                elif np.mean(gen_area)<min_treshold:
                    break     
                else:
                    pass
                count+=1
                if count>20:
                    print("Could not find suitable polynomial. Try again.")
                    break
                
            print("Chosen roots:",np.around(self.poly.roots,2))
            print("Chosen coefs:",np.around(self.poly.coefs,2))

            print("Done (RFA fractal initialized).")

    def init_array(self,N,domain):
        """create array of complex numbers"""
        real_dom=np.linspace(domain[0,0],domain[0,1],N,endpoint=False) #expanded domain
        complex_dom=np.linspace(domain[1,0],domain[1,1],N,endpoint=False)
        return np.array([(item+complex_dom*1j) for i,item in enumerate(real_dom)]).reshape(N,N).transpose() #array of shape (N,N)

    def Nova_Newton_method(self,array,func,dfunc,tol=1e-08,max_steps=100,damping=complex(1,1),c=0.15):
        """Newton method"""
        print("RFA-Newton method...",end="\r")
        #initialisation
        shape=array.shape
        z=array.flatten()
        
        prec=np.ones_like(z) #precision
        activepoint=np.ones_like(z)  #used to stop points that attained required precision
        i=0 #count, used to calculate the fractal
        ziter=np.zeros_like(z)

        while i<=max_steps:

            #checking for points precision reaching tol 
            e=np.where(prec<tol) #checking which points converged
            activepoint[e]=0 #taking out those points 

            ziter[e]=i #noting the count value of those points

            prec[e]=100 #taking out the precision at those points

            activeindex=activepoint.nonzero() #updating the active indexes
            
            #Newton Method
            dx=-damping*func(z)/dfunc(z)+c
            z[activeindex]=z[activeindex]+dx[activeindex]
            prec[activeindex]=abs(dx[activeindex])
            i+=1
            print("RFA-Newton method...",i,end="\r")
        #Assigning a value to points that haven't converged
        ziter[activepoint==True]=i
        z[activepoint==True]=0
        z=np.around(z,4)

        print("Done (RFA-Newton method) ")
    
        return ziter.reshape(shape),z.reshape(shape)

    def Nova_Halley_method(self,array,func,dfunc,d2func,tol=1e-08,max_steps=100,damping=complex(0.5,0.5),c=0.15):
        """Halley method"""

        print("RFA-Halley method...",end="\r")
        #initialisation
        shape=array.shape
        z=array.flatten()
        
        prec=np.ones_like(z) #precision
        activepoint=np.ones_like(z)  #used to stop points that attained required precision
        i=0 #count, used to calculate the fractal
        ziter=np.zeros_like(z)

        while i<=max_steps:

            #checking for points precision reaching tol 
            e=np.where(prec<tol) #checking which points converged
            activepoint[e]=0 #taking out those points 

            ziter[e]=i #noting the count value of those points

            prec[e]=100 #taking out the precision at those points

            activeindex=activepoint.nonzero() #updating the active indexes
            
            #Newton Method
            dx=-damping*2*func(z)*dfunc(z)/(2*dfunc(z)**2-d2func(z)*func(z))
            z[activeindex]=z[activeindex]+dx[activeindex]
            prec[activeindex]=abs(dx[activeindex])
            i+=1
            print("RFA-Halley method...",i,end="\r")
        #Assigning a value to points that haven't converged
        ziter[activepoint==True]=i
        z[activepoint==True]=0
        z=np.around(z,4)

        print("Done (RFA-Halley method) ")
    
        return ziter.reshape(shape),z.reshape(shape)

    #def Nova_Secant_method(self,tol=1.e-8,max_steps=100,damping_factor=complex(0.1,0.1),pixel=complex(0.1,0.1)):