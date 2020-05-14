#import numpy,scipy.sparse
#from sparsesvd import sparsesvd
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
#from scipy.sparse.linalg import svds, eigs
import scipy.misc
import math
from mpl_toolkits.mplot3d import Axes3D
import os
from picamera import PiCamera
from time import sleep
import io
from numpy import array

camera = PiCamera()

print('Begin capture')
frames = 64
for cap in range(0,frames+1):
    camera.resolution = (600,500)
    camera.start_preview()
    sleep(2)
    path = '/home/pi/Documents/PhotometricStereo/data/frame'+str(cap)+'.png'
    camera.capture(path)
    camera.stop_preview()
print('End capture')
