
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

    def Bairstow_method(self,epsilon=1.e-16,itermax=100):
        """Bairstow method"""
        print("RFA-Bairstow method")
        return 0


class RFA_Fractal_Nova():

    def __init__(self, param, **kwargs):
        self.__dict__.update(kwargs)



    #def Nova_Newton_method(self,epsilon=1.e-16,itermax=100):

       
