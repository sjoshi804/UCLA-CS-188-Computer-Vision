import scipy
import numpy as np
from matplotlib import pyplot as plt
from skimage.data import astronaut, moon
import math
from scipy import signal, misc
from numpy.linalg import inv

img = astronaut()
plt.figure()
plt.imshow(img)

print('The shape of the image is: {} '.format(np.shape(img)))

def rgb2gray(img):
    R=img[:,:,0]
    G=img[:,:,1]
    B=img[:,:,2]
    L = R*0.2989 + G*0.5870 + B*0.1140
    return L
img_gray =  rgb2gray(img)
plt.imshow(img_gray, cmap='gray')
print('The shape of the gray image is: {} '.format(np.shape(img_gray)))

def Gaussian(size,sigma):
  x,y = np.mgrid[(-size+1)/2:(size-1)/2+1,(-size+1)/2:(size-1)/2+1]
  gaussian_filter = ( 1 / (2 * np.pi * sigma*sigma) ) * np.exp( -(x*x + y*y) /(2 * sigma*sigma) )
  return gaussian_filter
gaussian_filter = Gaussian(7,4)
convolution = signal.convolve2d(img_gray,gaussian_filter,boundary='symm')
plt.imshow(convolution,cmap='gray')

def CONV_2D(image, kernel):
  m,n = kernel.shape
  x,y = image.shape
  convolution = np.zeros((x,y))
  padded_image = np.zeros((m+x+m-2,n+y+n-2))

  for i in range(x):
    for j in range(y):
      padded_image[m-1+i][n-1+j]=image[i][j]

  for i in range(x):
    for j in range(y):
      convolution[i][j] = np.sum(padded_image[m-1+i-m//2 : m+i+m//2, n-1+j-n//2 : n+j+n//2 ]*kernel)

  return convolution
      

convolution = CONV_2D(img_gray,gaussian_filter)
plt.imshow(convolution,cmap='gray')

horizontal = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
horizontal = CONV_2D(convolution,horizontal)
plt.imshow(horizontal,cmap='gray')

def grad(img):
  vertical = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  vertical = CONV_2D(convolution,vertical)
  gradient = np.hypot(horizontal,vertical)
  return gradient
gradient = grad(convolution)
plt.imshow(gradient,cmap='gray')

img=moon()
m,n = img.shape
def scale(k):
  return np.array([[k,0,0],[0,k,0],[0,0,1]])
def translation(x,y):
  return np.array([[1,0,x],[0,1,y],[0,0,1]])
def rotation(angle):
  return np.array([[np.cos(np.radians(angle)),-np.sin(np.radians(angle)),0],[np.sin(np.radians(angle)),np.cos(np.radians(angle)),0],[0,0,1]])


def transformation(image):
  transformed = np.zeros((m//2,n//2))
  X=[]
  Y=[]
  for i in range(m):
    for j in range(n):
        position=np.vstack((j,i,1))
        position = np.dot( translation((n//2)//2,(m//2)//2) , np.dot( rotation(180) , np.dot( translation((-n//2)//2,(-m//2)//2) , np.dot(scale(0.5),position) ) ) ).astype('uint8')
        X.append(position[0][0])
        Y.append(position[1][0])
        transformed[position[1][0]][position[0][0]] = image[i][j]

  return transformed
       
transformed_image = transformation(img)
plt.imshow(transformed_image,cmap='gray')


def interp_bilinear(Z, X, Y):

  H,W = np.shape(Z)

  X[X < 0] = 0
  X[X > W-1]= W-1
  Y[Y < 0] = 0
  Y[Y > H-1] = H-1

  f_val = []

  for x, y in zip(X, Y):

    y_min = math.floor(y)
    y_max = math.ceil(y)
    x_min = math.floor(x)
    x_max = math.ceil(x)

    x=x_max-x
    y=y_max-y

    f=Z[y_min,x_min]*(1-x)*(1-y)+Z[y_min,x_max]*y*(1-x)+Z[y_max,x_min]*(1-y)*x+Z[y_max,x_max]*x*y

    f_val.append(f)

  return f_val

