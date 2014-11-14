from PIL import Image
from numpy import *
from pylab import *
from PCV.tools import imtools, pca

path = '/home/duan/pycv/PCV-data/a_thumbs'
imlist = imtools.get_imlist(path)
im = array(Image.open(imlist[0]))
m, n = im.shape[0:2]
imnbr = len(imlist)

immatrix = array([ array(Image.open(im)).flatten() 
            for im in imlist ], 'f')

V, S, immean = pca.pca(immatrix)

figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(m,n))
for i in range(7):
    subplot(2,4,i+2)
    imshow(V[i].reshape(m,n))
show()


