
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import random
import time
from skimage.filters import threshold_mean
from scipy.optimize import root_scalar

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

            elif form=="coef":
                self.coefs=func
                self.roots=self.coefficients_to_roots(self.coefs)

            elif form=="taylor_approx":
                self.coefs=self.taylor_approximation(funcm,degree=self.__dict__.get("degree"),interval=self.__dict__.get("interval"))
                self.roots=self.coefficients_to_roots(self.coefs)


        ### POLYNOMIALS ###
    # Code taken from https://github.com/3b1b/videos/blob/master/_2022/quintic/roots_and_coefs.py

    def roots_to_coefficients(roots):
        """Convert roots form poly to coefficients form poly"""
    n = len(list(roots))
    return [
        ((-1)**(n - k)) * sum(
            np.prod(tup)
            for tup in it.combinations(roots, n - k)
        )
        for k in range(n)
    ] + [1]


    def poly(z, coefs):
        """Evaluate polynomial at z with coefficients coefs"""
        return sum(coefs[k] * z**k for k in range(len(coefs)))


    def dpoly(z, coefs):
        """Derivative of poly(z, coefs)"""
        return sum(k * coefs[k] * z**(k - 1) for k in range(1, len(coefs)))

    def d2poly(z,coefs):
        """Second derivative of poly(z, coefs)"""
        return sum(k*(k-1)*coefs[k]*z**(k-2) for k in range(2,len(coefs)))


    def find_root(func, dfunc, seed=complex(1, 1), tol=1e-8, max_steps=100):
        # Use newton's method
        last_seed = np.inf
        for n in range(max_steps):
            if abs(seed - last_seed) < tol:
                break
            last_seed = seed
            seed = seed - func(seed) / dfunc(seed)
        return seed


    def coefficients_to_roots(coefs):
        """Find roots of polynomial with coefficients coefs"""
        if len(coefs) == 0:
            return []
        elif coefs[-1] == 0:
            return coefficients_to_roots(coefs[:-1])
        roots = []
        # Find a root, divide out by (z - root), repeat
        for i in range(len(coefs) - 1):
            root = find_root(
                lambda z: poly(z, coefs),
                lambda z: dpoly(z, coefs),
            )
            roots.append(root)
            new_reversed_coefs, rem = np.polydiv(coefs[::-1], [1, -root])
            coefs = new_reversed_coefs[::-1]
        return roots

    def taylor_approximation(func,degree,interval):
        """Taylor approximation of function func"""
        if degree is None:
            degree=5
        if interval is None:
            interval=1
        
        from scipy.interpolate import approximate_taylor_polynomial

        return approximate_taylor_polynomial(func,0,degree,interval).c

    def choose_poly(max_degree):
        """Choose random polynomial"""
        if max_degree<2 or max_degree is None:
            print("max_degree must be >=2. Setting to 8")
            max_degree=8
        degree=random.randint(2,max_degree)
        coefs=[complex(random.uniform(-10,10),random.uniform(-10,10)) for i in range(degree)]
        return coefs


class RFA_Fractal():

    def __init__(self, config, **kwargs):
        
        self.config = config
        self.__dict__.update(kwargs)

        self.array = self.init_array(config["N"],config["domain"])

        self.poly = Polynomials(func=config["func"],random_poly=config["random"],form=config["form"],**kwargs)

        #check if poly converges ok with sample of domain
        if config["random"]==True
            while True:
                z=self.init_array(100,config["domain"])

                if config["method"]=="Newton":
                    up_treshold=0.35
                    min_treshold=0.12
                    z=self.Nova_Newton_method(lambda z: self.poly.poly,lambda z: self.poly.dpoly,z,tol=1.e-05,max_steps=50)

                else: #SAME FOR NOW
                    up_treshold=0.35
                    min_treshold=0.12
                    z=self.Nova_Newton_method(lambda z: self.poly.poly,lambda z: self.poly.dpoly,z,tol=1.e-05,max_steps=50)

                gen_area=z > threshold_mean(z)

                if np.mean(gen_area)>up_treshold:
                    break
                elif np.mean(gen_area)<min_treshold:
                    break     
                else:
                    pass
            print("Chosen roots",self.poly.roots)

    def init_array(self,N,domain):
        """create array of complex numbers"""
        real_dom=np.linspace(domain[0,0],domain[0,1],N,endpoint=False) #expanded domain
        complex_dom=np.linspace(domain[1,0],domain[1,1],N,endpoint=False)
        return np.array([(item+complex_dom*1j) for i,item in enumerate(real_dom)]).reshape(N,N).transpose() #array of shape (N,N)

    #def Nova_Newton_method(self,func,dfunc,tol=1.e-8,max_steps=100,damping_factor=complex(0.1,0.1),pixel=complex(0.1,0.1)):
        #flatten array

        #vectorize

        #iterate
            #count conv steps

        #unflatten

        #return array, conv_steps

    #def Nova_Halley_method(self,tol=1.e-8,max_steps=100,damping_factor=complex(0.1,0.1),pixel=complex(0.1,0.1)):

    #def Nova_Secant_method(self,tol=1.e-8,max_steps=100,damping_factor=complex(0.1,0.1),pixel=complex(0.1,0.1)):

class RFA_fractal():

    def __init__(self,parameters) -> None:
            print("Init Root-Finding-Algorithm (RFA-init) fractals...")
            ### EXTRACT PARAMETERS
            #polynomial coefficients
            self.degree=parameters["degree"] #degree of polynomial
            self.coord=parameters["coord"] #coordinates of computed complex plane
            self.size=parameters["dpi"] #Size of array
            self.array=self.init_array(self.size,parameters["coord"]).reshape(self.size,self.size) #array of complex numbers

            if parameters["rand_coef"]==True:
                self.choose_polynomial(self.degree,self.coord)
            else: self.coef=parameters["coef"]    
            print("Done RFA-Init")
    
    ## ARRAY
    def init_array(self,N,domain):
        """create array of complex numbers"""
        real_dom=np.linspace(domain[0,0],domain[0,1],N,endpoint=False) #expanded domain
        complex_dom=np.linspace(domain[1,0],domain[1,1],N,endpoint=False)
        return np.array([(item+complex_dom*1j) for i,item in enumerate(real_dom)]).reshape(N,N).transpose() #array of shape (N,N)

    ## POLYNOMIAL
    def Horner_method(self,x, a):
        result = 0
        for i in range(len(a)-1, -1, -1):
            result = a[i] + (x * result)
        return result
    
    def rand_coeffs(self,degree,a,b):
        """chooses coefficients for a polynomial of the given degree, such that f(a) == b"""
        coefficients = [0] + [random.randint(1, 10) for _ in range(degree-1)]
        y = sum(coefficient * a**n for n, coefficient in enumerate(coefficients))
        coefficients[0] = np.clip(b - y,-10000,10000)
        
        return coefficients

    def polynomial_compute(self,x,fp=False,fp2=False):
        """Return the value of a polynomial and its derivatives with determined coefficients at value x"""    
        #Value of the polynomial
        coefficients=self.coef
        f=self.Horner_method(x,coefficients)
        
        #Value of its first derivative

        coefficients_prime=[coefficients[i] * i for i in range(1, len(coefficients))]
        f_prime=self.Horner_method(x,coefficients_prime)                


        #Value of its second derivative

        coefficients_prime_second=[coefficients[i] * i for i in range(1, len(coefficients_prime))]
        f_prime_second=self.Horner_method(x,coefficients_prime_second)


        return f,f_prime,f_prime_second
    
    def choose_polynomial(self,degree,coord):
        print("Choosing polynomial..(RFA-choose_polynomial)")
        again=True
        while again:
            n=200 #small array 

            z=self.init_array(n,coord).reshape(n,n)
            self.coef =self.rand_coeffs(degree,random.randint(-2,2),random.randint(-2,2))
            
            itermax=150
            epsilon=1.e-06
            #initialisation
            prec=np.ones_like(z) #precision
            activepoint=np.ones_like(z)  #used to stop points that attained required precision
            dx=np.zeros_like(z) #step
            i=0 #count, used to calculate the fractal

            while np.any(activepoint==True) and i<=itermax:

                #checking for points precision reaching epsilon 
                e=np.where(prec<epsilon) #checking which points converged
                activepoint[e]=0 #taking out those points 
                z[e]=i #noting the count value of those points
                prec[e]=100 #taking out the precision at those points
                activeindex=activepoint.nonzero() #updating the active indexes
                
                #Newton Method
                fonction=self.polynomial_compute(z,fp=True) #f, fprime
                dx=-fonction[0]/fonction[1]
                z[activeindex]=z[activeindex]+dx[activeindex]
                prec[activeindex]=abs(dx[activeindex])
                i+=1
            #Assigning a value to points that haven't converged
            z[activepoint==True]=i

            gen_area=z > threshold_mean(z)

            if np.mean(gen_area)>0.35:
                again=True
            elif np.mean(gen_area)<0.12:
                again=True       
            else:
                again=False
        
        self.z=z
        print("Done (RFA-choose_polynomial)")
        return self.coef
    

    ## ROOT FINDING ALGORITHMS
    def Newton_method(self,epsilon=1.e-5,itermax=50):
        """Newton method"""
        print("RFA-Newton method...")
        #initialisation
        prec=np.ones_like(self.array) #precision
        activepoint=np.ones_like(self.array)  #used to stop points that attained required precision
        dx=np.zeros_like(self.array) #step
        i=0 #count, used to calculate the fractal
        z=self.array.copy()
        ziter=np.empty_like(self.array)

        while np.any(activepoint==True) and i<=itermax:

            #checking for points precision reaching epsilon 
            e=np.where(prec<epsilon) #checking which points converged
            activepoint[e]=0 #taking out those points 

            ziter[e]=i #noting the count value of those points

            prec[e]=100 #taking out the precision at those points
            activeindex=activepoint.nonzero() #updating the active indexes
            
            #Newton Method
            fonction=self.polynomial_compute(z,fp=True) #f, fprime
            dx=-fonction[0]/fonction[1]+0.15
            z[activeindex]=z[activeindex]+dx[activeindex]
            prec[activeindex]=abs(dx[activeindex])
            i+=1
            print(i,end="\r")
        #Assigning a value to points that haven't converged
        ziter[activepoint==True]=i
        z[activepoint==True]=0

        self.z=ziter
        z=np.around(z,4)

        print("Done (RFA-Newton method)")
        return ziter,z
    
    def Global_Newton_method(self,epsilon=1.e-16,itermax=100):
        """Global Newton method with damping factor a-> a/2"""
        print("RFA-Global Newton method")
        #initialisation
        prec=np.ones_like(self.array) #precision
        activepoint=np.ones_like(self.array)  #used to stop points that attained required precision
        dx=np.zeros_like(self.array) #step
        i=0 #count, used to calculate the fractal
        z=self.array.copy()

        a=1 #damping factor

        while np.any(activepoint==True) and i<=itermax:

            #checking for points precision reaching epsilon 
            e=np.where(prec<epsilon) #checking which points converged
            activepoint[e]=0 #taking out those points 
            z[e]=i #noting the count value of those points
            prec[e]=100 #taking out the precision at those points
            activeindex=activepoint.nonzero() #updating the active indexes
            
            #Newton Method
            fonction=self.polynomial_compute(z) #f, fprime
            dx=- fonction[0]/fonction[1]

            #correct damping factor
            Correction=abs(self.polynomial_compute(z[activeindex]+a*dx[activeindex],self.coef)[0])<abs(self.polynomial_compute(z[activeindex],self.coef)[0])
            while np.any(Correction):
                Correction=abs(self.polynomial_compute(z[activeindex]+a*dx[activeindex],self.coef)[0])<abs(self.polynomial_compute(z[activeindex],self.coef)[0])
                a/=2
            
            z[activeindex]=z[activeindex]+a*dx[activeindex]
            prec[activeindex]=abs(a*dx[activeindex])
            i+=1
        #Assigning a value to points that haven't converged
        z[activepoint==True]=i

        self.z=z
        print("Done (RFA-Gloabal Newton method)")
        return z

    def Halley_method(self,epsilon=1.e-16,itermax=50):
        """Global Newton method with damping factor a-> a/2"""
        print("RFA-Halley method")
        #initialisation
        prec=np.ones_like(self.array) #precision
        activepoint=np.ones_like(self.array)  #used to stop points that attained required precision
        dx=np.zeros_like(self.array) #step
        i=0 #count, used to calculate the fractal
        z=self.array.copy()
        ziter=np.empty_like(self.array)

        a=1+1/4j #damping factor

        while np.any(activepoint==True) and i<=itermax:

            #checking for points precision reaching epsilon 
            e=np.where(prec<epsilon) #checking which points converged
            activepoint[e]=0 #taking out those points 
            ziter[e]=i #noting the count value of those points
            prec[e]=100 #taking out the precision at those points
            activeindex=activepoint.nonzero() #updating the active indexes
            
            fonction=self.polynomial_compute(z,fp=True,fp2=True) #f, f', f''

            #Halley Method
            dx=- 2*fonction[0]*fonction[1]/(2*fonction[1]**2-fonction[2]*fonction[0])

            z[activeindex]=z[activeindex]+a*dx[activeindex]
            prec[activeindex]=abs(a*dx[activeindex])
            i+=1
            print(i,end="\r")
        #Assigning a value to points that haven't converged
        ziter[activepoint==True]=i
        z[activepoint==True]=0

        self.z=ziter
        z=np.around(z,4)

        print("Done (RFA-Halley method)")
        return ziter,z   

    def Secant_method(self,epsilon=1.e-16,itermax=100):
        """Secant method"""
        print("RFA-Secant method")
        return 0