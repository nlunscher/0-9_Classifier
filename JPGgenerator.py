# Create image jpgs from the formatted file

import numpy as np
import Image
import random

# read in the image Str dataset
def read_dataset(dataset_file_name):
	im_strs = []
	with open(dataset_file_name) as f:
		im_strs = f.readlines()

	return im_strs

# determine image label and remove label from data
def determine_label(imStr):
	im_split = imStr.split(' ')

	im_labels = np.zeros(10)
	str_labels = im_split[-11:-1]
	for i in range(10):
		im_labels[i] = float(str_labels[i])

	# print im_labels

	return np.argmax(im_labels)

# read the bits and reshape it to the image
def format_image(imStr):
	im_dim = 16

	im_split = (imStr.split(' '))[:-11]
	im_split_float = np.zeros(len(im_split))

	for i in range(len(im_split)):
		im_split_float[i] = float(im_split[i])

	im_reshape = im_split_float.reshape(im_dim, im_dim) * 255
	return im_reshape

# create and save the image from a shape of floats
def create_jpg(im_float, fileName):
	im_out = Image.fromarray(im_float.astype('uint8')).convert('L')
	im_out.save(fileName)


# write to a file (for as a refernce for Caffe to read the images)
def writeTextLine(file_name, line_text):
	with open(file_name, 'a') as f:
		f.write(line_text + '\n')

# search through the database and generate the JPGs
def generate_JPGs(data_source, im_folder, im_ref_file, digits_wanted, percent_test):
	im_strs = read_dataset(data_source)

	im_totals = np.zeros(10)
	im_wanted = [[],[],[],[],[],[],[],[],[],[]]
	for i in range(len(im_strs)):
		im_label = determine_label(im_strs[i])

		im_totals[im_label] += 1

		if im_label in digits_wanted:
			im_wanted[im_label].append(i)

	# create 2 lists, one of test one of train
	im_test = []
	im_train = []
	for digit in digits_wanted:
		d_num = 0
		d_total = im_totals[digit]
		n_test = np.ceil(d_total * percent_test)

		for i in im_wanted[digit]:
			d_num += 1
			if d_num <= n_test:
				im_test.append(i)
			else:
				im_train.append(i)
	random.shuffle(im_test)
	random.shuffle(im_train)

	im_num = 0
	for i in im_test:
		im_num += 1
		im_float = format_image(im_strs[i])		
		im_label = determine_label(im_strs[i])

		file_name = str(im_num) + '_' + str(im_label) + 'image.jpg'
		folder = im_folder + 'test/' 
		ref_file = im_ref_file + '_test.txt'
		create_jpg(im_float, folder + file_name)
		writeTextLine(folder + ref_file, folder + file_name + ' ' + str(im_label))
	
	for i in im_train:
		im_num += 1
		im_float = format_image(im_strs[i])		
		im_label = determine_label(im_strs[i])

		file_name = str(im_num) + '_' + str(im_label) + 'image.jpg'
		folder = im_folder + 'train/' 
		ref_file = im_ref_file + '_train.txt'
		create_jpg(im_float, folder + file_name)
		writeTextLine(folder + ref_file, folder + file_name + ' ' + str(im_label))

	return im_totals

image_folder = '0-9_images/'
dataset_file_name = image_folder + 'SemeionHandwrittenDataset.txt'
im_ref_file_name = 'im_reference'
percent_test = 0.10 # of each label
digits_wanted = [0,1,2,3,4,5,6,7,8,9]

im_totals = generate_JPGs(dataset_file_name, image_folder, 
							im_ref_file_name, digits_wanted, percent_test)
print im_totals



print
print
print "================ JPGs Generator done ================"