# Using the Classifier

#### Setup environment
from pylab import *
import matplotlib.pyplot as plt
import numpy as np

# add caffe to python path
import sys
caffe_root = '../../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Setup model
caffe.set_mode_cpu()

#### Very important to implement this deploy stuff
# https://github.com/BVLC/caffe/wiki/Using-a-Trained-Network:-Deploy
##### Re place the last layer (soft max with loss) with this softmax one
# layer {
#   name: "loss"
#   type: "Softmax"
#   bottom: "prob"
#   top: "loss"
# }
##### Also this one
# layer {
#   name: "data"
#   type: "Input"
#   top: "data"
#   input_param { 
#     shape: { dim: 1 dim: 3 dim: 16 dim: 16 }
#   }
# }

model_folder = 'Model_Files/'
model_name = 'zero9_net'
model_def = model_folder + model_name + '_deploy.prototxt'
model_weights = model_folder + model_name + '.caffemodel'

net = caffe.Net(model_def,		#defines the structure of the mdoel
				model_weights,	# contains the trained weights
				caffe.TEST)		# use test mode (e.g. dont perform dropout) - leave out nodes during testinf to help prevent overfitting

print
print "Model Loaded"
print



transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)


# visualize the input data
def visualize_data(data):
	new_data = np.zeros((16,16))

	mx = np.amax(data)

	for x in range(len(data[0][0][0])):
		for y in range(len(data[0][0])):
			 av_data = np.mean([data[0][0][y][x], data[0][1][y][x], data[0][2][y][x]])
			 new_data[y][x] = 1 if av_data > mx*.9 else 0
	print new_data



# setup the data you want to use
# image_file = '0-9_images/test/33_7image.jpg'
# image_file = '0-9_images/test/3_9image.jpg'
# image_file = '0-9_images/test/112_6image.jpg'
image_file = '0-9_images/test/131_5image.jpg'

image = caffe.io.load_image(image_file)

# classify the image
net.blobs['data'].data[...] = transformer.preprocess('data', image) # copy image to memory allocated for the net
output = net.forward() # the actual classification
predict = net.blobs['loss'].data.argmax(1)

print "Data: "
print visualize_data(net.blobs['data'].data[...])

print 'Predicted class is: ', predict
print 'Raw output: '
print output['loss'][0]


print
print
print "=================== 01_Classifier_CNN_Using ==================="
