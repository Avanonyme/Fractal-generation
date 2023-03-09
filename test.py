import numpy as np
import matplotlib.pyplot as plt

# Define the Newton's method equation and its derivative
def f(z):
    return z**10 - 1

def f_prime(z):
    return 10*z**9

# Define the traps to look for
def trap_circle(z, r):
    return abs(z) < r

def trap_star(z, r):
    x, y = z.real, z.imag
    return abs(x**3 - 3*x*y**2) + abs(3*x**2*y - y**3) < r

# Define the number of iterations and the size of the image
max_iters = 50
size = 400

# Create a grid of complex numbers
x = np.linspace(-2, 2, size)
y = np.linspace(-2, 2, size)
c = x[:,np.newaxis] + 1j*y[np.newaxis,:]

# Initialize the image as a 2D array of zeros
image = np.zeros((size, size))

# Loop over each point in the grid
for i in range(size):
    for j in range(size):
        z = c[i,j]
        
        # Iterate the Newton's method equation until convergence or max_iters
        for k in range(max_iters):
            z = z - f(z) / f_prime(z)
            
            # Keep track of the last few points in the orbit
            if k >= max_iters - 20:
                if trap_circle(z, 0.1):
                    # Assign a color based on the circle trap
                    image[i,j] = 1
                    break
                elif trap_star(z, 0.05):
                    # Assign a color based on the star trap
                    image[i,j] = 2
                    break
        else:
            # Assign a default color based on convergence speed
            image[i,j] = k
print(len(np.where(image==1)),len(np.where(image==2)))
# Display the image with a color map
plt.imshow(image, cmap='jet')
plt.show()
