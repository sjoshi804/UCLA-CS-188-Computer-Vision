import scipy
from scipy import misc
import numpy as np
from matplotlib import pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import rotate
from skimage.transform import radon
import pdb

img = shepp_logan_phantom()

plt.figure()
plt.imshow(img, cmap='gray')

print('The shape of the phantom image is: {} '.format(np.shape(img)))

rotated_image=rotate(img,90)

plt.figure()
plt.imshow(rotated_image, cmap='gray')
plt.show()

num_projections=len(img)
theta=np.linspace(0,180,num_projections)
sinogram=radon(img,theta=theta)
plt.imshow(sinogram,cmap='gray')

array = np.zeros((len(img),len(img)))
for i in range(len(img)):
  rotated_image = rotate(img, -i*180/400)
  array[:,i]=sum(rotated_image)
plt.imshow(array,cmap="gray")

m, n = sinogram.shape
array = np.zeros((m,n))
w = np.linspace(-np.pi,np.pi,n)
absolute=np.abs(w)
filtered_response=np.fft.fftshift(absolute)
result = np.zeros((m,n))
for i in range(len(absolute)):

  fourier = np.fft.fft(sinogram[:,i])
  projection = np.multiply(fourier,filtered_response)
  inverse = np.fft.ifft(projection)
  result[:,i]=np.real(inverse)

plt.imshow(result,cmap="gray")

m,n = result.shape
mid_index = (m)// 2
reconstructed = np.zeros((m,n))
[Y,X] = np.mgrid[0:m,0:n]
xpr = X-mid_index
ypr = Y-mid_index
theta = np.linspace(0,180,n,endpoint=False)
th = (np.pi/180)*theta
for i in range(len(th)):
  cur_points = mid_index +xpr*np.cos(th[i]) - ypr*np.sin(th[i])
  spot = np.where((cur_points>0)&(cur_points<n))
  cur_reconstructed = np.zeros((m,n))
  new_points=cur_points[spot]
  new_points = new_points.astype('int32')
  cur_reconstructed[spot]=result[new_points,i]
  reconstructed = reconstructed+cur_reconstructed

reconstructed = reconstructed*(np.pi)/(2*len(th))

plt.imshow(reconstructed,cmap="gray")
plt.show()
