'''
PROJECT 2 - STEREO RECONSTRUCTION
Hayleigh Sanders RIN# 661195735
Computer Vision - ECSE 6650
10/18/2019
---Setup Instructions for Windows---
Run this in Python 3.7 - NOT 3.8! (package install will not work in Windows 10)
Use git bash to install packages. Download from: https://git-scm.com/downloads
Run this command in git bash: pip install scipy numpy matplotlib
'''
import numpy as np
import random
import math
import scipy
from scipy.linalg import null_space
from scipy.linalg import svd
from numpy.linalg import matrix_rank
from numpy import linalg as LA
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#initialize data arrays
Lp2d = np.empty((0,2),int)
Rp2d = np.empty((0,2),int)
p3d = np.empty((0,3),int)
lfpts = np.empty((0,2),int)
rfpts = np.empty((0,2),int)

#populate arrays with data
f=open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project2\project2_data\calibration\pts_2D_left.txt", "r")
f1 = f.readlines()
for x in f1:
    q = x.split()
    q[1] = int(float(q[1]))
    q[2] = int(float(q[2]))
    Lp2d = np.append(Lp2d,[[int(q[1]),int(q[2])]],axis=0)
f.close()

f=open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project2\project2_data\calibration\pts_2D_right.txt", "r")
f2 = f.readlines()
for x in f2:
    q=x.split()
    q[1] = int(float(q[1]))
    q[2] = int(float(q[2]))
    Rp2d = np.append(Rp2d,[[int(q[1]),int(q[2])]],axis=0)
f.close()

f=open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project2\project2_data\calibration\pts_3D.txt", "r")
f3 = f.readlines()
for x in f3:
    q=x.split()
    q[1] = int(float(q[1]))
    q[2] = int(float(q[2]))
    q[3] = int(float(q[3]))
    p3d = np.append(p3d,[[int(q[1]),int(q[2]),int(q[3])]],axis=0)
f.close()

f=open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project2\project2_data\faceimage\pts_left.txt", "r")
f4 = f.readlines()
for x in f4:
    q = x.split()
    q[1] = int(float(q[1]))
    q[2] = int(float(q[2]))
    lfpts = np.append(lfpts,[[int(q[1]),int(q[2])]],axis=0)
f.close()

f=open(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project2\project2_data\faceimage\pts_right.txt", "r")
f4 = f.readlines()
for x in f4:
    q = x.split()
    q[1] = int(float(q[1]))
    q[2] = int(float(q[2]))
    rfpts = np.append(rfpts,[[int(q[1]),int(q[2])]],axis=0)
f.close()

#given a subset of points r, generate a projection matrix
#Return a list of 12 p matrix values
def get_p(p2d, p3d, r):
    #Form the A matrix
    A = np.empty((0,12),float)
    for i in r:
        M1 = p3d[i][0]
        M2 = p3d[i][1]
        M3 = p3d[i][2]
        
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

    #Recover camera parameters from P
    c0=p11*p31+p12*p32+p13*p33
    r0=p21*p31+p22*p32+p23*p33
    #print ("c0,r0: ",c0,r0)

    r31=p31
    r32=p32
    r33=p33

    sxf = math.sqrt(p11*p11+p12*p12+p13*p13 - c0*c0)
    syf = math.sqrt(p21*p21+p22*p22+p23*p23 - r0*r0)
    #print ("sxf,syf: ",sxf,syf)

    tz = p34
    tx = (p14 - c0*tz)/(sxf)
    ty = (p24 - r0*tz)/(syf)
    #print ("t: ",tx,ty,tz)

    r11 = (p11-c0*r31)/(sxf)
    r12 = (p12-c0*r32)/(sxf)
    r13 = (p13-c0*r33)/(sxf)
    #print ("r1: ",r11,r12,r13)

    r21 = (p21-c0*r31)/(syf)
    r22 = (p22-c0*r32)/(syf)
    r23 = (p23-c0*r33)/(syf)
    #print ("r2: ",r21,r22,r23)
    #print ("r3: ",r31,r32,r33)
    
    W = np.array([[sxf,0,c0],[0,syf,r0],[0,0,1]])
    R = np.array([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
    T = np.array([tx,ty,tz])
    P = np.array([[p11,p12,p13,p14],[p21,p22,p23,p24],[p31,p32,p33,p34]])
    
    return (W,R,T)

def eight_point(Lp2d,Rp2d,r):
    #Form the A matrix
    A = np.empty((0,9),float)
    for i in r:
        cl = Lp2d[i][0]
        rl = Lp2d[i][1]
        cr = Rp2d[i][0]
        rr = Rp2d[i][1]
        a = [cl*cr, rl*cr, cr, cl*rr, rl*rr, rr, cl, rl, 1]
        A = np.append(A,[a],axis=0)
        
    u, s, vh = svd(A)
    u.shape, s.shape, vh.shape
    V = vh[8]
    f1 = [V[0],V[1],V[2]]
    f2 = [V[3],V[4],V[5]]
    f3 = [V[6],V[7],V[8]]
    F = np.array([f1,f2,f3])
    return F

def get_perror(W,R,T,p2d,p3d):
    sxf = W[0][0]
    syf = W[1][1]
    c0 = W[0][2]
    r0 = W[1][2]
    r11=R[0][0]
    r12=R[0][1]
    r13=R[0][2]
    r21=R[1][0]
    r22=R[1][1]
    r23=R[1][2]
    r31=R[2][0]
    r32=R[2][1]
    r33=R[2][2]
    tx=T[0]
    ty=T[1]
    tz=T[2]
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

    p_error=0
    in_e = []
    for x in range(0,47):
        a = abs(new_p2d[x][0] - p2d[x][0]) #change in x
        b = abs(new_p2d[x][1] - p2d[x][1]) #change in y
        d = math.sqrt(a*a + b*b)
    p_error = p_error+d
    return (p_error,new_p2d)
    #print ("Average projection error: ",p_error/48)

def plot_2d(new_p2d, p2d, img):
    im = plt.imread(img)
    implot = plt.imshow(im)
    x = new_p2d[:,0]
    y = new_p2d[:,1]
    x2 = p2d[:,0]
    y2 = p2d[:,1]
    plt.scatter(x,y,c='b')
    plt.scatter(x2,y2,c='r')
    plt.show()    

#create an array of k non-repeating random integers
def get_random(k):
    rand = []
    for i in range(k):
        r=random.randint(0,47)
        if r not in rand: rand.append(r)
    return rand

def w_scale(W,x,y):
    W[0][0] = W[0][0]*x
    W[1][1] = W[1][1]*y
    return W

def reconstruct3D(cl,rl,cr,rr,Pl,Pr):
    
    a = (cl*(Pl[2])-Pl[0])
    b = (rl*(Pl[2])-Pl[1])
    c = (cr*(Pr[2])-Pr[0])
    d = (rr*(Pr[2])-Pr[1])
    A = np.array([a,b,c,d])
    '''
    A = np.array([
        [(Pl[0][0]-Pl[2][0]*cl), (Pl[0][1]-Pl[2][1]*cl), (Pl[0][2]-Pl[2][2]*cl), 0, (Pl[0][3]-Pl[2][3]*cl)],
        [(Pl[1][0]-Pl[2][0]*rl), (Pl[1][1]-Pl[2][1]*rl), (Pl[1][2]-Pl[2][2]*rl), 0, (Pl[1][3]-Pl[2][3]*rl)],
        [0,0,0,0,0],
        [(Pr[0][0]-Pr[2][0]*cr), (Pr[0][1]-Pr[2][1]*cr), (Pr[0][2]-Pr[2][2]*cr), (Pr[0][3]-Pr[2][3]*cr),0],
        [(Pr[1][0]-Pr[2][0]*rr), (Pr[1][1]-Pr[2][1]*rr), (Pr[1][2]-Pr[2][2]*rr), (Pr[1][3]-Pr[2][3]*rr),0],
        [0,0,0,0,0]])
    '''
    #print (A)
    u, s, vh = svd(A)
    u.shape, s.shape, vh.shape
    #print (vh[3])
    ns = vh[3]
    #print (ns)
    return [ns[0]/ns[3],ns[1]/ns[3],ns[2]/ns[3]]

#Find F with the eight point algorithm
rand = get_random(8)
Fep = eight_point(Lp2d,Rp2d,rand)
print ("F from the eight point algorithm:")
print (Fep)

#Calibrate cameras separately

rand = get_random(46)
LP = get_p(Lp2d,p3d,rand)
rand = get_random(46)
RP = get_p(Lp2d,p3d,rand)

Wl = LP[0]
Rl = LP[1]
Tl = LP[2]

Wr = RP[0]
Rr = RP[1]
Tr = RP[2]


#Get relative R
RrT = Rr.transpose()
R = Rl @ RrT

#Get relative T
RTr = R @ Tr
T = Tl - RTr

#Find F by separate calibration
S = np.array([[0,-1*T[2],T[1]],[T[2],0,-1*T[0]],[-1*T[1],T[0],0]])
St = S.transpose()
E = St @ R

Wri = np.linalg.inv(Wr)
Wli = np.linalg.inv(Wl)
Wlti = Wli.transpose()
F = Wlti @ E @ Wri
print ("F from individual camera calibration:")
print (F)

#Find epipoles
u, s, vh = svd(F)
u.shape, s.shape, vh.shape
er = vh[2]
er = er/er[2]
print ("Right epipole: ")
print (er)

Ft = F.transpose()
u, s, vh = svd(Ft)
u.shape, s.shape, vh.shape
el = vh[2]
el = el/el[2]
print ("Left epipole: ")
print (el)

#Find T/(||T||)
TTnorm = (T/(np.linalg.norm(T)))
Tx = TTnorm[0]
Ty = TTnorm[1]
Tz = TTnorm[2]

#Find Rectified R for the left view
e1 = np.array([Tx,Ty,Tz])
e2 = np.array([Ty,(-1*Tx),0])
tconst = 1/(math.sqrt(Tx*Tx + Ty*Ty))
e2 = tconst*e2
e3 = np.cross(e1,e2)
Rrect = np.array([e1,e2,e3])

#Adjust sampling on fsx, fsy
sx = 1
sy = 1
Wl = w_scale(Wl,sx,sy)

#find right view transform
Wlinv = np.linalg.inv(Wl)
Rrect_r = Rrect @ R
Rrect_r_t = Rrect_r.transpose()
Wr = w_scale(Wr,sx,sy)
Wrinv = np.linalg.inv(Wr)
Tformr = Wr @ Rrect_r_t @ Wrinv

#find left view transform
Rrectt = Rrect.transpose()
Tform = Wl @ Rrectt @ Wlinv

lface = imread(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project2\project2_data\faceimage\left_face.jpg")
rface = imread(r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project2\project2_data\faceimage\right_face.jpg")

#plot left epipolar lines on original image
axes = plt.gca()
axes.set_xlim([-5000,5000])
axes.set_ylim([-5000,5000])
implot = plt.imshow(lface)
for pts in lfpts:
    Ul = [pts[0],pts[1],1]
    ell = F @ Ul
    a = ell[0]
    b = ell[1]
    c = ell[2]
    slope = -1*(a/b)
    intercept = c/b
    plt.plot([pts[0],(-1*intercept/slope)],[pts[1],0],'k-')
plt.show()

#plot right epipolar lines on original image
axes = plt.gca()
axes.set_xlim([-5000,5000])
axes.set_ylim([-5000,5000])
implot = plt.imshow(rface)
for pts in rfpts:
    Ur = [pts[0],pts[1],1]
    elr = F @ Ur
    a = elr[0]
    b = elr[1]
    c = elr[2]
    slope = -1*a/b
    intercept = c/b
    plt.plot([pts[0],(-1*intercept/slope)],[pts[1],0],'k-')
plt.show()

rect_lfpts = np.empty((0,2),int)
rect_rfpts = np.empty((0,2),int)

fwd_ltfm = Wl @ Rrect @ Wlinv
fwd_rtfm = Wr @ Rrect_r @ Wrinv
    
hl = lface.shape[0]
wl = lface.shape[1]
rect_lface = np.empty((hl, wl, 3), dtype=np.uint8)
for c in range(0,wl):
    for r in range(0,hl):
        q = np.array([c,r,1])
        z = Tform @ q
        cnew = z[0]
        rnew = z[1]
        lnew = z[2]
        cnew = int(cnew/lnew)
        rnew = int(rnew/lnew)
        try:
            rect_lface[r][c] = lface[cnew][rnew]
        except IndexError:
            pass

#find rectified face points on left image
for pts in lfpts:
    Ul = [pts[0],pts[1],1]
    new_ul = fwd_ltfm @ Ul
    new_ul = new_ul/new_ul[2]
    q = [new_ul[0],new_ul[1]]
    rect_lfpts = np.append(rect_lfpts,[q],axis=0)

#plot left epipolar lines on rectified image
axes = plt.gca()
axes.set_xlim([-5000,5000])
axes.set_ylim([-5000,5000])
implot = plt.imshow(rect_lface)
for pts in rect_lfpts:
    Ul = [pts[0],pts[1],1]
    ell = F @ Ul
    a = ell[0]
    b = ell[1]
    c = ell[2]
    slope = -1*a/b
    intercept = c/b
    plt.plot([pts[0],0],[pts[1],intercept],'k-')
plt.show()
#rect_lface = np.fliplr(rect_lface)
#rect_lface = np.rot90(rect_lface, k=1, axes=(1,0))
hr = rface.shape[0]
wr = rface.shape[1]
rect_rface = np.empty((hr+1000, wr+1000, 3), dtype=np.uint8)
for c in range(0,wr):
    for r in range(0,hr):
        q = np.array([c,r,1])
        z = Tformr @ q
        cnew = z[0]
        rnew = z[1]
        lnew = z[2]
        cnew = int(cnew/lnew)
        rnew = int(rnew/lnew)
        try:
            rect_rface[c+500][r+500] = rface[rnew][cnew]
        except IndexError:
            pass
#find rectified face points on right image
for pts in rfpts:
    Ur = [pts[0],pts[1],1]
    new_ur = fwd_rtfm @ Ul
    new_ur = new_ur/new_ul[2]
    q = [new_ur[0],new_ur[1]]
    rect_rfpts = np.append(rect_rfpts,[q],axis=0)

#plot right epipolar lines on rectified image
axes = plt.gca()
axes.set_xlim([-5000,5000])
axes.set_ylim([-5000,5000])
implot = plt.imshow(rect_rface)
for pts in rect_rfpts:
    Ur = [pts[0],pts[1],1]
    elr = F @ Ur
    a = elr[0]
    b = elr[1]
    c = elr[2]
    slope = -1*a/b
    intercept = c/b
    plt.plot([pts[0],0],[pts[1],intercept],'k-')
plt.show()

rt_l = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
Pl = Wl @ rt_l
rt_r = np.array([[R[0][0],R[0][1],R[0][2],T[0]],
                [R[1][0],R[1][1],R[1][2],T[1]],
                [R[2][0],R[2][1],R[2][2],T[2]]])
Pr = Wr @ rt_r
x = []
y = []
z = []
allpts = np.empty((0,3),int)
for i in range(0,29):
    pt3d = reconstruct3D(lfpts[i][0],lfpts[i][1],rfpts[i][0],rfpts[i][1],Pl,Pr)
    #print (pt3d)
    x.append(pt3d[0])
    y.append(pt3d[1])
    z.append(pt3d[2])
    q = [pt3d[0],pt3d[1],pt3d[2]]
    allpts = np.append(allpts,[q],axis=0)

np.savetxt(r'C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project2\project2_data\output.txt', allpts, delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')    
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.001)
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.savefig('teste.pdf')
plt.show()

plt.imshow(rect_lface)
plt.show()

plt.imshow(rect_rface)
plt.show()


