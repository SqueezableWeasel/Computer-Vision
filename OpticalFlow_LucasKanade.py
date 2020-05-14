'''
Hayleigh Sanders
11/20/2019
Homework 6: Optical Flow
Run with Python 3.7.5
'''
import numpy as np
from numpy import array
from PIL import Image
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
import math

#Lucas-Kanade algorithm
#Input: two images of equal size, pixel neighborhood size N
#thin = number of pixels between each plotted motion vector (others don't get plotted)
#thinning value is added because the quiver graph can be dense and visualizing optical
#flow can be difficult if all the vectors are plotted
def optical_flow(im1, im2, N, thin):
    #Filters for finding the image gradients
    x_filter = np.array([[-1., 1.],
                         [-1., 1.]])
    
    y_filter = np.array([[-1., -1.],
                         [1., 1.]])
    
    t_filter = np.array([[1., 1.],
                         [1., 1.]])
    
    bound = int(N/2) #compute boundary around center pixel

    #find the image gradients by convolving the filters with the images
    x_gradient = signal.convolve2d(im1, x_filter, boundary='symm', mode='same')
    y_gradient = signal.convolve2d(im1, y_filter, boundary='symm', mode='same')
    t_gradient = signal.convolve2d(im2, t_filter, boundary='symm', mode='same') + signal.convolve2d(im1, -1*t_filter, boundary='symm', mode='same')

    #make u,v arrays
    u = np.zeros((im1.shape[0],im1.shape[1]))
    v = np.zeros((im1.shape[0],im1.shape[1]))

    fig, ax = plt.subplots() #set up plot
    plt.imshow(im2) #add the second image to the plot
    for x in range(bound, im1.shape[0]-bound): #iterate through the NxN window
        for y in range(bound, im1.shape[1]-bound):
            Ix = x_gradient[x-bound:x+bound+1, y-bound:y+bound+1].flatten() #find Ix
            Iy = y_gradient[x-bound:x+bound+1, y-bound:y+bound+1].flatten() #find Iy
            It = t_gradient[x-bound:x+bound+1, y-bound:y+bound+1].flatten() #find It
            b = np.reshape(It, (It.shape[0],1)) #find b
            A = np.vstack((Ix, Iy)) #form the A matrix
            A = A.transpose()
            #Make sure the smallest eigenvalue is larger than a tiny threshold value - we don't want to plot
            #vectors for pixels with near-singular values (close to zero movement)
            Vxy = np.linalg.pinv(A) @ b #(A^t * A)^-1 * A^t is the pseudoinverse of A - multiply with b to get v(x,y)
            u[x,y]=Vxy[0]
            v[x,y]=Vxy[1]
            #if the optical flow magnitude is low or the condition of A is large, set the optical flow at this point to zero
            if math.sqrt(Vxy[0]**2 + Vxy[1]**2) >= .001 or np.linalg.cond(A) < 6:
                if y%thin == 0 and x%thin == 0: #plot every multiple of the thin value to reduce plot overcrowding 
                    q = ax.quiver(y,x,Vxy[0],-1*Vxy[1],width=.002,color='red') #plot a quiver for v(x,y) at point x,y (V is inverted because the image vertical axis is inverted)
            else:
                u[x,y]=0
                v[x,y]=0
    plt.show()
    return (u,v)

#Load images, convert to grayscale
s11 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq1\sphere2.pgm").convert('L')
s12 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq1\sphere3.pgm").convert('L')
s13 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq1\sphere4.pgm").convert('L')
s14 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq1\sphere5.pgm").convert('L')
s15 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq1\sphere6.pgm").convert('L')

s21 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq2\center1.jpg").convert('L')
s22 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq2\center2.jpg").convert('L')
s23 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq2\center3.jpg").convert('L')
s24 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq2\center4.jpg").convert('L')
s25 = Image.open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj3_seq2\center5.jpg").convert('L')

#convert images to numpy arrays
s11 = array(s11)
s12 = array(s12)
s13 = array(s13)
s14 = array(s14)
s15 = array(s15)

s21 = array(s21)
s22 = array(s22)
s23 = array(s23)
s24 = array(s24)
s25 = array(s25)

#perform optical flow algorithm on images
optical_flow(s11,s12,20,5)
optical_flow(s12,s13,20,5)
optical_flow(s13,s14,20,5)
optical_flow(s14,s15,20,5)

optical_flow(s21,s22,5,25)
optical_flow(s22,s23,5,25)
optical_flow(s23,s24,5,25)
optical_flow(s24,s25,5,25)

