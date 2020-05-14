'''
Final Project - Uncalibrated photometric stereo
Hayleigh Sanders RIN#661195735
'''
import numpy,scipy.sparse
from sparsesvd import sparsesvd
import scipy
from scipy.linalg import svd
from numpy.linalg import matrix_rank
import numpy as np
from numpy import linalg as LA
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
import scipy as sp
import scipy.sparse
import scipy.misc
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from numpy import array

#load image data into an array
nameroot = "Image_"
images = []
for i in range(1,64):
    '''
    if i<10:
        stri = "0"+str(i)
    else:
        stri = str(i)
    '''
    #imname = r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\PhotometricStereo\Images_Cat\Image_"+stri+".png"
    imname = r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\PhotometricStereo\Brick64\frame"+str(i)+".png"    
    img = Image.open(imname).convert('L')
    #img.show()
    img = array(img)
    img = img/255
    imsize = img.shape
    images.append(img)

imlength = images[0].shape[0]*images[0].shape[1]
nimages = len(images)
I = np.empty((0,imlength))

#form image data matrix
for img in images:
    i = img.flatten()
    I = np.append(I,[i],axis=0)

#Reject shadow data
thresh = 0
temp = I.T
for x in range(0,I.T.shape[0]):
    z = 0
    c = 0
    for yy in range(0,I.T.shape[1]):
        if I.T[x][yy] > thresh:
            c = c+1
            z = z+I.T[x][yy]
    avg = sum(temp[x])/temp[x].shape
    for y in range(0,I.T.shape[1]):
        if I.T[x][y] < thresh:
            temp[x][y] = c/z
I = temp.T

#Perform SVD on I
(U,s,vh) = np.linalg.svd(I, full_matrices=False)

#find the rank-three approximations of the SVD components
L = U[:,0:3]
N = vh[0:3,:]
S_sqrt = np.diag(np.sqrt(s[:3]))
L = np.dot(L, S_sqrt)
N = np.dot(S_sqrt, N)
l = None
#print (L)

#Form a system of equations to solve for B components
for i in range(0, L.shape[0]):
    x = L[i,0]
    y = L[i,1]
    z = L[i,2]
    row = [x*x, 2*x*y, 2*x*z, y*y, 2*y*z, z*z]

    if l is None:
        l = np.array(row)
    else:
        l = np.vstack((l,row))
(b,res,rank,s) = np.linalg.lstsq(l,np.ones(L.shape[0]),rcond=None)

#Form symmetric matrix B
B = np.array([[b[0], b[1], b[2]],
              [b[1], b[3], b[4]],
              [b[2], b[4], b[5]]])

#Decompose B via SVD to find A
(U,s,vh) = np.linalg.svd(B)
A = np.dot(U,np.diag(np.sqrt(s)))

#find L and N
L = np.dot(L,A)
N = np.linalg.solve(A,N)
#for x in L:
#    print(math.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]))
#print(N)

i=0
#initialize data matrices
normalsx = np.zeros((images[0].shape))
normalsy = np.zeros((images[0].shape))
normalsz = np.zeros((images[0].shape))
normal = np.zeros((images[0].shape))
thresh = np.zeros((images[0].shape[0], images[0].shape[1],3),dtype='uint8')

height = np.zeros((images[0].shape))

hx = np.zeros((images[0].shape))
hy = np.zeros((images[0].shape))

normalrgb = np.zeros((images[0].shape[0], images[0].shape[1],3),dtype='uint8')
normalxyz = np.zeros((images[0].shape[0], images[0].shape[1],3),dtype='uint8')

albedo = np.zeros((images[0].shape[0],images[0].shape[1],3))

fig = plt.figure()
ax = fig.gca(projection='3d')

ic = []

#print(normals)
imx = images[0].shape[0]
imy = images[0].shape[1]

for x in range(0,imx):
    for y in range(0,imy):
        norm = N[:,i]
        #to save computational time, only consider non-zero normals (they are already initialized to zero in the matrices about to be populated)
        if not np.isnan(np.sum(norm)):
            #print(norm)

            normalsx[x][y] = norm[0]
            normalsy[x][y] = norm[1]
            normalsz[x][y] = norm[2]
            normalxyz[x][y] = norm
            '''
            t=.00001
            if norm[0] > t and norm[1] > t and norm[2] > t:
                thresh[x][y] = [abs(norm[0]*160), abs(norm[1]*160), abs(norm[2]*160)]
            '''
            normal[x][y] = math.sqrt(norm[0]*norm[0]+norm[1]*norm[1]+norm[2]*norm[2])
            normalrgb[x][y] = [abs(norm[0]*160), abs(norm[1]*160), abs(norm[2]*160)]

            #Find surface heights
            diff=0
            if x>0 and y>0:
                diff = ((normalsx[x][y]-normalsx[x][y-1]) - (normalsy[x][y]-normalsy[x-1][y]))**2
            #print (diff)
            ic.append(diff)
            dely = 0
            delx = 0
            if diff < .001:
                for c in range(1,x):
                    delx = delx + (normalsx[c][y]) - (normalsx[c-1][y])
                for r in range(1,y):
                    dely = dely + (normalsy[x][r]) - (normalsy[x][r-1])
                height[x][y] = delx+dely

            ic.append(((normalsx[x][y]) - (normalsy[x][y]))**2)
            
            '''
            if x>0 and y>0:
                height[x][y] = height[x-1][y-1]+(normalsx[x][y]/normalsz[x][y])+(normalsy[x][y]/normalsz[x][y])
            '''
            #plot normals on a quiver plot
            if x%5 ==0 and y%5 == 0:
                q = ax.quiver(x,y,z,norm[0]*10,norm[1]*10,norm[2]*10,length=.01) #plot a quiver for v(x,y) at point x,y (V is inverted because the image vertical axis is inverted)
            
        i = i+1
#normalrgb = Idiag @ normalrgb
#print(max(diff))
ax.view_init(90, 0)
plt.show()

#ax.view_init(90, 0)
#plt.show()

#print(normalxyz[0][0])

#plt.imshow(normal*150)
#plt.show()
#print(normal)
#alb = Image.fromarray(L)

alb = Image.fromarray(normal*160)
rgb = Image.fromarray(normalrgb,'RGB')
shadow = Image.fromarray(thresh,'RGB')
dmap = Image.fromarray(height*160)

dmap.show()
alb.show()
#shadow.show()
rgb.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(range(0,images[0].shape[1]), range(0,images[0].shape[0]))

#surf = ax.plot_surface(X, Y, height/150, cmap=cm.gray, linewidth=0, antialiased=False)
#fig = go.Figure(data=[go.Mesh3d(X, Y, height/150, color='gray', opacity=0.50)])
#fig.show()
#ax.contour3D(X, Y, height/150, 500,cmap='gray')
ax.plot_surface(X, Y, height/150, rstride=5, cstride=5, cmap='gray', edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(90, 90)
plt.show()




