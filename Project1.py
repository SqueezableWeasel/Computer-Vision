'''
Hayleigh Sanders
10/18/2019
RIN# 661195735
Run this in Python 2.7
'''
import numpy as np
import random
import math
from scipy.linalg import null_space
from scipy.linalg import svd
from numpy.linalg import matrix_rank
from numpy import linalg as LA
from operator import itemgetter
import matplotlib.pyplot as plt

#initialize data arrays
p2d = np.empty((0,2),float)
p3d = np.empty((0,3),float)
bad_p3d = np.empty((0,3),float)

#populate arrays with data
f=open("Left_2Dpoints.txt", "r")
f1 = f.readlines()
for x in f1:
    q = x.split()
    q[0] = float(q[0])
    q[1] = float(q[1])
    p2d = np.append(p2d,[q],axis=0)
f.close()

f=open("bad_3dpts.txt", "r")
f2 = f.readlines()
for x in f2:
    q=x.split()
    q[0] = float(q[0])
    q[1] = float(q[1])
    q[2] = float(q[2])
    bad_p3d = np.append(bad_p3d,[q],axis=0)
f.close()

f=open("3Dpointnew.txt", "r")
f3 = f.readlines()
for x in f3:
    q=x.split()
    q[0] = float(q[0])
    q[1] = float(q[1])
    q[2] = float(q[2])
    p3d = np.append(p3d,[q],axis=0)
f.close()

#given a subset of points r, generate a projection matrix
#Return a list of 12 p matrix values
def get_p(r):
    #Form the A matrix
    A = np.empty((0,12),float)
    for i in r:
        M1 = bad_p3d[i][0]
        M2 = bad_p3d[i][1]
        M3 = bad_p3d[i][2]
        
        c = p2d[i][0]
        r = p2d[i][1]

        a = [M1,M2,M3,1,0,0,0,0, -1*c*M1, -1*c*M2, -1*c*M3, -1*c]
        b = [0,0,0,0,M1,M2,M3,1, -1*r*M1, -1*r*M2, -1*r*M3, -1*r]

        A = np.append(A,[a],axis=0)
        A = np.append(A,[b],axis=0)
    #uncomment to display matrix rank
    #print "Rank: ",matrix_rank(A)
    #Perform SVD on A - take the last row of Vh - corresponding to the last column of S^t
    #Vh is actually the S matrix (this library named it differently)
    u, s, vh = svd(A)
    u.shape, s.shape, vh.shape
    V = vh[11]
    #uncomment to display nullspace
    #print "Nullspace: \n", null_space(A)
    #print "V: \n", V
    p11 = V[0]
    p12 = V[1]
    p13 = V[2]
    p14 = V[3]

    p21 = V[4]
    p22 = V[5]
    p23 = V[6]
    p24 = V[7]

    p31 = V[8]
    p32 = V[9]
    p33 = V[10]
    p34 = V[11]

    #compute scale factor
    sf = math.sqrt(1/(p31*p31+p32*p32+p33*p33))
    #Scale p values
    p11=p11*sf
    p12=p12*sf
    p13=p13*sf
    p14=p14*sf

    p21=p21*sf
    p22=p22*sf
    p23=p23*sf
    p24=p24*sf

    p31=p31*sf
    p32=p32*sf
    p33=p33*sf
    p34=p34*sf
    return (p11,p12,p13,p14,p21,p22,p23,p24,p31,p32,p33,p34)

#generate a list of 2d points given a projection matrix p
#and subset of points r; return array of 2D projected points
def _2dprojection(p,r):
    p11=p[0]
    p12=p[1]
    p13=p[2]
    p14=p[3]
    p21=p[4]
    p22=p[5]
    p23=p[6]
    p24=p[7]
    p31=p[8]
    p32=p[9]
    p33=p[10]
    p34=p[11]
    #Recover camera parameters from P
    c0=p11*p31+p12*p32+p13*p33
    r0=p21*p31+p22*p32+p23*p33
    print ("c0,r0: ",c0,r0)

    r31=p31
    r32=p32
    r33=p33

    sxf = math.sqrt(p11*p11+p12*p12+p13*p13 - c0*c0)
    syf = math.sqrt(p21*p21+p22*p22+p23*p23 - r0*r0)
    print ("sxf,syf: ",sxf,syf)

    tz = p34
    tx = (p14 - c0*tz)/(sxf)
    ty = (p24 - r0*tz)/(syf)
    print ("t: ",tx,ty,tz)

    r11 = (p11-c0*r31)/(sxf)
    r12 = (p12-c0*r32)/(sxf)
    r13 = (p13-c0*r33)/(sxf)
    print ("r1: ",r11,r12,r13)

    r21 = (p21-c0*r31)/(syf)
    r22 = (p22-c0*r32)/(syf)
    r23 = (p23-c0*r33)/(syf)
    print ("r2: ",r21,r22,r23)
    print ("r3: ",r31,r32,r33)

    new_p2d = np.empty((0,2),float)
    p_error = 0
    for pts in p3d:
        #Compute 2D points from projection parameters
        x=pts[0]
        y=pts[1]
        z=pts[2]
        c = (sxf)*((r11*x+r12*y+r13*z+tx)/(r31*x+r32*y+r33*z+tz))+c0
        r = (syf)*((r21*x+r22*y+r23*z+ty)/(r31*x+r32*y+r33*z+tz))+r0
        q = (c,r)
        new_p2d = np.append(new_p2d,[q],axis=0)
    return new_p2d

#create an array of k non-repeating random integers
def get_random(k):
    rand = []
    for i in range(k):
        r=random.randint(0,71)
        if r not in rand: rand.append(r)
    return rand

results = []
########STEP 1
print ("--------STEP 1--------")
rand = get_random(7)
#rand=[36, 31, 24, 25, 71, 59, 51]
#for x in rand:
#    print p3d[x],"->",bad_p3d[x]
P = get_p(rand)
print ("P: ",P)
new_p2d = _2dprojection(P,rand)

inliers1 = []
p_error=0
for x in range(0,71):
    a = abs(new_p2d[x][0] - p2d[x][0]) #change in x
    b = abs(new_p2d[x][1] - p2d[x][1]) #change in y
    d = math.sqrt(a*a + b*b)
    if d<5:
        inliers1.append(x)
    p_error = p_error+d
print ("Average projection error: ",p_error/72)
print ("Inliers at: ",inliers1)
print ("Number of inliers: ",len(inliers1))
results.append(((p_error/72),new_p2d,P))

########STEP 2
print ("--------STEP 2--------")
rand = get_random(7)
P2 = get_p(rand)
print ("P2: ",P2)
new_p2d_2 = _2dprojection(P2,rand)

inliers2 = []
p_error=0
for x in range(0,71):
    a = abs(new_p2d_2[x][0] - p2d[x][0]) #change in x
    b = abs(new_p2d_2[x][1] - p2d[x][1]) #change in y
    d = math.sqrt(a*a + b*b)
    if d<5:
        inliers2.append(x)
    p_error = p_error+d
print ("Average projection error: ",p_error/72)
print ("Inliers at: ",inliers2)
print ("Number of inliers: ",len(inliers2))

#estimate new projection error
p_error=0
in_e = []
for x in range(0,71):
    a = abs(new_p2d_2[x][0] - p2d[x][0]) #change in x
    b = abs(new_p2d_2[x][1] - p2d[x][1]) #change in y
    d = math.sqrt(a*a + b*b)
    if d<5:
        inliers2.append(x)
    p_error = p_error+d
print ("Average projection error: ",p_error/72)
results.append(((p_error/72),new_p2d_2,P2))

#select the P matrix with more inliers
results = sorted(results,key=itemgetter(0))
ransac_p2 = results[0][1]

#Estimate 2D points from P
x = p2d[:,0]
y = p2d[:,1]
x2 = new_p2d_2[:,0]
y2 = new_p2d_2[:,1]
x3 = ransac_p2[:,0]
y3 = ransac_p2[:,1]
#Plot RANSAC and linear 2D data on the calibration pattern
im = plt.imread("frame1.bmp")
implot = plt.imshow(im, cmap='gray')
plt.scatter(x,y,c='b')
plt.scatter(x2,y2,c='g')
plt.scatter(x3,y3,c='r')
plt.show()
