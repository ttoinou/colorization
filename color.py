import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.color as color
import scipy.ndimage.interpolation as sni
import caffe
from PIL import Image

parser = argparse.ArgumentParser(description='Colorization')
parser.add_argument('--dir', '-d', type=str)
parser.add_argument('--input', '-i', default='', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
parser.add_argument('--rebal', dest='rebalance', action='store_true')
parser.add_argument('--norebal', dest='rebalance', action='store_false')
parser.set_defaults(rebalance=True)

args = parser.parse_args()

# %matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)
caffe.set_mode_gpu()
caffe.set_device(0)

# Select desired model
if args.rebalance:
	net = caffe.Net('models/colorization_deploy_v2.prototxt', 'models/colorization_release_v2.caffemodel', caffe.TEST)
else:
	net = caffe.Net('models/colorization_deploy_v2.prototxt', 'models/colorization_release_v2_norebal.caffemodel', caffe.TEST)
# net = caffe.Net('../models/colorization_deploy_v1.prototxt', '../models/colorization_release_v1.caffemodel', caffe.TEST)
# If you are training your own network, you may replace the *.caffemodel path with your trained network.

(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape

print 'Input dimensions: (%i,%i)'%(H_in,W_in)
print 'Output dimensions: (%i,%i)'%(H_out,W_out)

pts_in_hull = np.load('resources/pts_in_hull.npy') # load cluster centers
net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel
print 'Annealed-Mean Parameters populated'

# load the original image
def colorize(urlIn,urlOut):
	img_rgb = caffe.io.load_image(urlIn)

	img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
	img_l = img_lab[:,:,0] # pull out L channel
	(H_orig,W_orig) = img_rgb.shape[:2] # original image size

	# create grayscale version of image (just for displaying)
	img_lab_bw = img_lab.copy()
	img_lab_bw[:,:,1:] = 0
	img_rgb_bw = color.lab2rgb(img_lab_bw)

	# resize image to network input size
	img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
	img_lab_rs = color.rgb2lab(img_rs)
	img_l_rs = img_lab_rs[:,:,0]

	# show original image, along with grayscale input to the network
	img_pad = np.ones((H_orig,W_orig/10,3))
	plt.imshow(np.hstack((img_rgb, img_pad, img_rgb_bw)))
	plt.title('(Left) Loaded image   /   (Right) Grayscale input to network')
	plt.axis('off');

	net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
	net.forward() # run network

	ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
	ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
	img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
	img_rgb_out = np.clip(color.lab2rgb(img_lab_out)*255,0,255) # convert back to rgb

	plt.imshow(img_rgb_out);
	plt.axis('off');

	result = np.uint8(img_rgb_out)
	Image.fromarray(result).save(urlOut)

if len(args.input) > 1:
	colorize(args.input,args.out)
elif os.path.isdir(args.dir):
	fs = os.listdir(args.dir)
        imagesPaths = []
        for fn in fs:
            base, ext = os.path.splitext(fn)
            if ext == '.jpg' or ext == '.png':
                imagepath = os.path.join(args.dir,fn)
                imagesPaths.append(imagepath)
        
        print 'folder ',args.input,' has ',len(imagesPaths),'images'
        
        for imagePath in imagesPaths:
            colorize(imagePath,os.path.join(args.out,os.path.basename(imagePath)))

'''
Copied from notebook :

import numpy as np
import argparse
from PIL import Image
import time

import chainer
from chainer import cuda, Variable, serializers
from net import *

parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('input')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--model', '-m', default='models/style.model', type=str)
parser.add_argument('--out', '-o', default='out.jpg', type=str)
args = parser.parse_args()

model = FastStyleNet()
serializers.load_npz(args.model, model)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

start = time.time()
image = xp.asarray(Image.open(args.input).convert('RGB'), dtype=xp.float32).transpose(2, 0, 1)
image = image.reshape((1,) + image.shape)
x = Variable(image)


y = model(x)
result = cuda.to_cpu(y.data)

result = result.transpose(0, 2, 3, 1)
result = result.reshape((result.shape[1:]))
result = np.uint8(result)
print time.time() - start, 'sec'

Image.fromarray(result).save(args.out)
'''
