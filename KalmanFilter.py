'''
ECSE 6650 Project 3
Hayleigh Sanders RIN# 661195735
Project 3 - Motion
12/8/2019
Run with Python 3.7.5
'''

import numpy as np
from numpy import array
from PIL import Image
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.image import imread
import math
from mpl_toolkits.mplot3d import Axes3D

#populate an array with image data
images = []
for i in range(10,40):
    imname = r"C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\proj4_img_seq\_"+str(i)+".pgm"
    img = imread(imname)
    imsize = img.shape
    images.append(img)

#show image 1 and 2 to manually identify polyhedron vertices
#imgplot = plt.imshow(images[0],cmap='gray')
#plt.show()
#imgplot = plt.imshow(images[2],cmap='gray')
#plt.show()

#Manually identify feature points for images 1 and 2
f1 = np.array([[118,78],
               [117,89],
               [122,158],
               [232,92],
               [254,80],
               [273,79],
               [273,163],
               [297,69],
               [296,138]])

f2 = np.array([[115,78],
               [114,89],
               [120,158],
               [229,91],
               [251,80],
               [270,78],
               [270,163],
               [295,69],
               [295,138]])

s0 = np.empty((0,4))
for x in range(0,f1.shape[0]):
    vx = f2[x][0] - f1[x][0]
    vy = f2[x][1] - f1[x][1]
    q = [f2[x][0],f2[x][1],vx,vy]
    s0 = np.append(s0,[q],axis=0)

#allpts = np.empty(0,(0,4))
W = np.empty((0,56))
data = np.empty((0,17))
alltraces = np.empty((0,28))
#for each feature point, initialize a kalman filter
for point in range(0,9):
    #print(s0[point])
    #initialize constant matrices
    phi = np.array([[1,0,1,0],
                    [0,1,0,1],
                    [0,0,1,0],
                    [0,0,0,1]])

    H = np.array([[1,0,0,0],
                  [0,1,0,0]])

    Q = np.array([[16,0,0,0],
                  [0,16,0,0],
                  [0,0,4,0],
                  [0,0,0,4]])

    R = np.array([[4,0],
                  [0,4]])

    Cov = np.array([[100,0,0,0],
                    [0,100,0,0],
                    [0,0,25,0],
                    [0,0,0,25]])
    pts = np.empty((0,4))
    traces = []
    xcords = []
    ycords = []
    xsum = 0
    ysum = 0
    st = s0[point]
    #print("Point: ",point)
    #cycle through each frame with the kalman filter
    for img in range(2,30):
        #print(Cov)
        sp = phi @ st #state prediction
        Covp = phi @ Cov @ phi.T + Q #covariance prediction
        subCovp = Covp[0:2,0:2] #2x2 submatrix of covariance prediction matrix

        #print("st: \n", st,"\n")
        #print("st+1: \n", sp,"\n")
        #print("Cov: \n", Cov,"\n")
        #print("Cov t+1: \n", Covp,"\n")
        #print("2x2 submatrix: \n",Covp[0:2,0:2])
        #print("Covp trace: ",np.trace(Cov))
        
        #px = int(sp[0])
        #py = int(sp[1])
        px = int(st[0])
        py = int(st[1])
        stx = int(st[0])
        sty = int(st[1])
        #print(images[2].shape)
        #print("intensity at s1: ",images[img-1][sty][stx])
        target = images[img-1][sty][stx]
        #print("intensity at predicted s2: ",images[img][py][px])
        bound = 9 #9 for pt 2
        #print(np.linalg.eigvals(subCovp))
        
        search = images[img][py-bound:py+bound+1,px-bound:px+bound+1]
        #print(search)
        #print(math.sqrt(Covp[0]))

        #pts = np.zeros((0,3))

        l = [images[img][py][px],py,px]
        diff = 300
        for x in range(px-bound,px+bound+1):
            for y in range(py-bound,py+bound+1):
                #q = [abs(images[img][y][x]-target), x, y]
                #pts = np.append(pts,[q],axis=0)
                dist = math.sqrt((px-x)**2+(py-y)**2)
                #print("distance: \n",dist,"\n")
                if images[img][y][x] < diff and dist < 6: #6 for pt 2
                    diff = images[img][y][x]
                    l = [images[img][y][x],x,y]
        #print(l)
        #pts = np.sort(pts,axis=-1)
        #print(pts)
        xnew = l[1]
        ynew = l[2]
        z = [xnew,ynew]
        #print("Measured: \n",z,"\n")
        #print("Intensity at z: \n",images[img][int(ynew)][int(xnew)],"\n")

        #print(H,subCovp,H.T,R)
        subk = H @ Covp @ H.T + R
        subk = np.linalg.inv(subk)
        K = Covp @ H.T @ subk #gain matrix
        #print("K: \n",K,"\n")
        pse = st + K @ (z - (H @ st)) #posterior state estimation
        pse = [pse[0],pse[1],pse[0]-st[0],pse[1]-st[1]]
        #print(pse)
        xsum = xsum+pse[0]
        ysum = ysum+pse[1]
        
        xcords.append(pse[0])
        ycords.append(pse[1])
        dat = [img+1,pse[0],pse[1],pse[2],pse[3],Cov[0][0],Cov[0][1],Cov[0][2],Cov[0][3],Cov[1][0],Cov[1][1],Cov[1][2],Cov[1][3],Cov[2][0],Cov[2][1],Cov[2][2],Cov[2][3]]
        data = np.append(data,[dat],axis=0)
        #print('Frame#',img+1,'X:',pse[0],'Y:',pse[1],'Vx:',pse[2],'Vy:',pse[3],'\nCovariance Matrix:\n',Cov[0],'\n',Cov[1],'\n',Cov[2],'\n')
        pts = np.append(pts,[pse],axis=0)
        pce = (np.identity(H.shape[1]) - K @ H) @ Covp #posterior covariance estimation
        #print("posterior covariance estimation: \n",pce,"\n")
        #print("PCE trace: ",np.trace(pce))
        #print("__________")
        traces.append(np.trace(pce))
        st = pse
        Cov = pce
        
        #plt.plot(pse[0],pse[1],marker='o',color='red')
        #imgplot = plt.imshow(images[img],cmap='gray')
        #plt.show()
        
    #print("__________")
    #print('w:',w)
    alltraces = np.append(alltraces,[traces],axis=0)
    xsum = xsum / (len(xcords))
    ysum = ysum / (len(ycords))
    w = []
    #Form W matrix
    for e in range(0,len(xcords)):
        c_ = xcords[e] - xsum
        r_ = ycords[e] - ysum
        w.append(c_)
        w.append(r_)
    W = np.append(W,[w],axis=0)
    
    #print("Point #",point,": ")
    print(pts)
    #np.savetxt(r'C:\Users\Hayleigh Sanders\Documents\ECSE6650\ECSE6650_Project3\output.csv', data, delimiter=',')
    
    #allpts.append[pts]
    #print(traces)

alltraces = alltraces.T
avgs = []
for rows in alltraces:
    avgs.append(sum(rows)/9)
print(avgs)

#plot trace vs frames
#plt.style.use('seaborn-whitegrid')
#frames = list(range(2,30))
#plt.plot(frames, avgs, '-ok')
#plt.show()

W = W.T
u,s,vh = np.linalg.svd(W)
'''
U_ = u[:,0:3]
S_sqrt = np.diag(np.sqrt(s[:3]))
V_ = vh[0:3,:]

R_ = U_ @ S_sqrt
S_ = S_sqrt @ V_

for rr in range(0,56,2):
    r11 = R_[rr][0]
    r12 = R_[rr][1]
    r13 = R_[rr][2]

    r21 = R_[rr+1][0]
    r22 = R_[rr+1][1]
    r23 = R_[rr+1][2]

    rrr = np.array([[r11**2, 2*r11*r12, 2*r11*r13, r12**2, 2*r12*r13, r13**2],
                    [r21**2, 2*r21*r22, 2*r22*r23, r22**2, 2*r22*r23, r23**2],
                    [r11*r21, r12*r21+r11*r22, r13*r21+r11*r23, r12*r22, r13*r22+r12*r23, r13*r23]])

    iii = np.array([[1,1],
                   [1,1],
                   [0,1]])
    A = np.linalg.lstsq(rrr,iii,rcond=None)
    print(A[0])
    A = A[0]
    AA = np.array([[A[0][0], A[1][0], A[2][0]],
                   [A[1][0],A[3][0], A[4][0]],
                   [A[2][0],A[4][0],A[5][0]]])
    print(AA)
    print(rr)
    print(np.linalg.cholesky(AA))
    #print(R_[rr], R_[rr+1])

V_ = vh[0:3,:]

'''

L = u[:,0:3]
N = vh[0:3,:]

S_sqrt = np.diag(np.sqrt(s[:3]))
L = np.dot(L, S_sqrt)
N = np.dot(S_sqrt, N)

L_help = None
#print (L)
for i in range(0, L.shape[0]):
    x = L[i,0]
    y = L[i,1]
    z = L[i,2]
    arr = [x*x, 2*x*y, 2*x*z, y*y, 2*y*z, z*z]

    if L_help is None:
        L_help = np.array(arr)

    else:
        L_help = np.vstack((L_help,arr))

(b_p,res,rank,s) = np.linalg.lstsq(L_help,np.ones(L.shape[0]),rcond=None)

B = np.array([[b_p[0], b_p[1], b_p[2]],
              [b_p[1], b_p[3], b_p[4]],
              [b_p[2], b_p[4], b_p[5]]])

(U,s,vh) = np.linalg.svd(B)
A = np.dot(U,np.diag(np.sqrt(s)))

L = np.dot(L,A)
N = np.linalg.solve(A,N)

Rot = np.array([[1,  0, 0],
		[0, 1, 0],
		[0, 0, 1]])

L = np.dot(L,Rot)
N = np.linalg.solve(Rot,N)

print(L)
print(N.T)
#imgplot = plt.imshow(images[0],cmap='gray')
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(N[0], N[1], N[2], c='r', marker='o')

plt.show()




