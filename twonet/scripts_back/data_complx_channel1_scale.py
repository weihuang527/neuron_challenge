from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import torch
import time
import numpy as np
import random
import torchvision
from PIL import Image
import tifffile
import multiprocessing
from joblib import Parallel
from joblib import delayed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class Train(Dataset):
	def __init__(self, cfg):
		super(Train, self).__init__()
		# multiprocess settings
		num_cores = multiprocessing.cpu_count()
		self.parallel = Parallel(n_jobs=num_cores, backend='threading')
		self.use_mp = False
		self.cfg = cfg

		# basic settings
		self.folder_name = cfg.DATA.folder_name
		self.data_list = cfg.DATA.train_data_list
		self.crop_size = list(cfg.DATA.patch_size)
		self.invalid_border = cfg.DATA.invalid_border
		self.scale_range = cfg.DATA.scale_range
		
		# simple augmentations
		self.random_fliplr = cfg.DATA.AUG.random_fliplr
		self.random_flipud = cfg.DATA.AUG.random_flipud
		self.random_flipz = cfg.DATA.AUG.random_flipz
		self.random_rotation = cfg.DATA.AUG.random_rotation
		
		# color augmentations
		self.color_jitter = cfg.DATA.AUG.color_jitter
		self.brightness = cfg.DATA.AUG.COLOR.brightness
		self.contrast = cfg.DATA.AUG.COLOR.contrast
		self.saturation = cfg.DATA.AUG.COLOR.saturation
		
		# gauss noise
		self.gauss_noise = cfg.DATA.AUG.gauss_noise
		self.gauss_mean = cfg.DATA.AUG.GAUSS.gauss_mean
		self.gauss_sigma = cfg.DATA.AUG.GAUSS.gauss_sigma

		# elastic transform
		self.elastic_trans = cfg.DATA.AUG.elastic_trans
		self.alpha_range = cfg.DATA.AUG.ELASTIC.alpha_range
		self.sigma = cfg.DATA.AUG.ELASTIC.sigma
		self.shave = cfg.DATA.AUG.ELASTIC.shave
		
		# extend crop size
		self.crop_size[0] = self.crop_size[0] + 2 * self.shave if self.elastic_trans else self.crop_size[0]
		self.crop_size[1] = self.crop_size[1] + 2 * self.shave if self.elastic_trans else self.crop_size[1]
		
		# color jitter
		self.cj = torchvision.transforms.ColorJitter(self.brightness, self.contrast, self.saturation, hue=0)

		# read train data
		f_list = open(os.path.join(self.folder_name, self.data_list), 'r')
		self.path_list = [x[:-1] for x in f_list.readlines()]
		assert len(self.path_list[0].split(' ')) == 2, 'Data list error!'
		self.num = len(self.path_list)
	
	def __getitem__(self, index):
		# random crop
		s = random.randint(-self.scale_range, self.scale_range)
		crop_size_x = self.crop_size[0] + s
		crop_size_y = self.crop_size[1] + s

		k = random.randint(0, self.num - 1)
		files = self.path_list[k]
		name = files.split(' ')
		raw_name = name[0]
		label_name = name[1]
		raw = np.asarray(cv2.imread(os.path.join(self.folder_name, raw_name), cv2.IMREAD_GRAYSCALE))
		label = np.asarray(cv2.imread(os.path.join(self.folder_name, label_name), cv2.IMREAD_GRAYSCALE))

		i = random.randint(0, raw.shape[0] - crop_size_x)
		j = random.randint(0, raw.shape[1] - crop_size_y)
		im = raw[i:i+crop_size_x, j:j+crop_size_y]
		lb = label[i:i+crop_size_x, j:j+crop_size_y]

		# scale
		im = cv2.resize(im, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_LINEAR)
		lb = cv2.resize(lb, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)
		
		# combine (3, C, H, W)
		# im = im.transpose(2, 0, 1)
		im = im[np.newaxis, :, :]
		lb = lb[np.newaxis, :, :]
		im_lb = np.concatenate([im, lb], axis=0)
		# im = np.expand_dims(im, axis=0)
		# im_lb = np.concatenate([im, lb], axis=0)
		
		# random flip
		if self.random_fliplr and random.uniform(0, 1) < 0.5:
			for j in range(im_lb.shape[0]): im_lb[j, :, :] = np.fliplr(im_lb[j, :, :])
			# im_lb = self._fliplr(im_lb)
		if self.random_flipud and random.uniform(0, 1) < 0.5:
			for j in range(im_lb.shape[0]): im_lb[j, :, :] = np.flipud(im_lb[j, :, :])
			# im_lb = self._flipud(im_lb)
		if self.random_flipz and random.uniform(0, 1) < 0.5:
			for j in range(im_lb.shape[0]): im_lb[j, :, :] = np.transpose(im_lb[j, :, :])
			# im_lb = np.flip(im_lb, axis=1)
		
		# random rotation
		if self.random_rotation:
			r = random.randint(0, 3)
			# if r: im_lb = self._rotate(im_lb, r)
			for j in range(im_lb.shape[0]): im_lb[j, :, :] = np.rot90(im_lb[j, :, :], r)
		
		# split (1/2, C, H, W)
		# im, lb = np.split(im_lb, [1])
		# # im = np.squeeze(im, axis=0)
		# lb = np.squeeze(lb, axis=0)
		im = im_lb[0]
		lb = im_lb[1]
		
		# random brightness, contrast and saturation
		if self.color_jitter:
			im = self._color_jitter(im)
		
		if self.gauss_noise:
			im = self._gauss_noise(im)
		
		""" debug 
		im = im.copy(); lb = lb.copy()
		self._draw_grid(im[0, :, :], gray_level=255)
		self._draw_grid(lb[1, 0, :, :], gray_level=0) """
		
		# elastic transform
		if self.elastic_trans:
			im, lb = self._elastic_transform(im, lb)
		#Image.fromarray(0.9 * im[1, :, :] + 0.1 * lb[1, 1, :, :]).show()
		
		im = im.astype(np.float32) / 255.0
		lb = lb.astype(np.float32) // 255
		im = im[np.newaxis, :, :]
		lb = lb[np.newaxis, :, :]

		return im, lb
	
	def __len__(self):
		return int(sys.maxsize)
	
	@staticmethod
	def _shave(im, border):
		if len(im.shape) == 4:
			return im[:, :, border[0] : -border[0], border[1] : -border[1]]
		elif len(im.shape) == 3:
			return im[:, border[0] : -border[0], border[1] : -border[1]]
		elif len(im.shape) == 2:
			return im[border[0] : -border[0], border[1] : -border[1]]
		else:
			raise NotImplementedError
	
	@staticmethod
	def _pad_with(vector, pad_width, iaxis, kwargs):
		pad_value = kwargs.get('padder', 10)
		vector[:pad_width[0]] = pad_value
		vector[-pad_width[1]:] = pad_value
		return vector
	
	def _fliplr(self, im_lb):
		outputs = []
		for sub_vol in im_lb:
			results = []
			for input in sub_vol:
				results.append(np.expand_dims(np.fliplr(input), axis=0))
			outputs.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
		return np.concatenate(outputs, axis=0)
	
	def _flipud(self, im_lb):
		outputs = []
		for sub_vol in im_lb:
			results = []
			for input in sub_vol:
				results.append(np.expand_dims(np.flipud(input), axis=0))
			outputs.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
		return np.concatenate(outputs, axis=0)
	
	def _rotate(self, im_lb, r):
		outputs = []
		for sub_vol in im_lb:
			results = []
			for input in sub_vol:
				results.append(np.expand_dims(np.rot90(input, r), axis=0))
			outputs.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
		return np.concatenate(outputs, axis=0)
	
	def _color_jitter(self, im):
		# im = im.transpose([1,2,0])
		new_im = np.asarray(self.cj(Image.fromarray(im)))
		# new_im = new_im.transpose([2,0,1])
		return new_im
		# return np.asarray(self.cj(Image.fromarray(im)))
		# results = []
		# for input in im:
		# 	results.append(np.expand_dims(np.asarray(self.cj(Image.fromarray(input))), axis=0))
		# return np.concatenate(results, axis=0)
	
	@staticmethod
	def _draw_grid(im, grid_size=50, gray_level=255):
		for i in range(0, im.shape[1], grid_size):
			cv2.line(im, (i, 0), (i, im.shape[0]), color=(gray_level,))
		for j in range(0, im.shape[0], grid_size):
			cv2.line(im, (0, j), (im.shape[1], j), color=(gray_level,))
	
	@staticmethod
	def _map(input, indices, shape):
		return np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0)
	
	def _gauss_noise(self, img):
		img = img.astype(np.float32) / 255.0
		noise = np.random.normal(self.gauss_mean, self.gauss_sigma ** 0.5, img.shape)
		out = img + noise
		if out.min() < 0:
			low_clip = -1.
		else:
			low_clip = 0.
		out = np.clip(out, low_clip, 1.0)
		out = (out * 255).astype(np.uint8)
		return out
	
	def _elastic_transform(self, image_in, label_in, random_state=None):
		"""Elastic deformation of image_ins as described in [Simard2003]_.
		.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
		   Convolutional Neural Networks applied to Visual Document Analysis", in
		   Proc. of the International Conference on Document Analysis and
		   Recognition, 2003.
		"""
		alpha = np.random.uniform(0, self.alpha_range)
		
		if random_state is None:
			random_state = np.random.RandomState(None)
		
		# shape = image_in.shape[1:]
		shape = image_in.shape
		
		dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode='constant', cval=0) * alpha
		dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode='constant', cval=0) * alpha
		
		x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
		indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
		
		if self.use_mp:
			image_out = np.concatenate(self.parallel(delayed(self._map)(input, indices, shape) for input in image_in), axis=0)
		else:
			image_out = map_coordinates(image_in, indices, order=1).reshape(shape)
			# image_out = []
			# for input in image_in:
			# 	image_out.append(np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0))
			# image_out = np.concatenate(image_out, axis=0)
			# image_out = []
			# for input in image_in:
			# 	image_out.append(np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0))
			# image_out = np.concatenate(image_out, axis=0)
		
		if self.use_mp:
			label_out = []
			for sub_vol in label_in:
				results = np.concatenate(self.parallel(delayed(self._map)(input, indices, shape) for input in sub_vol), axis=0)
				label_out.append(np.expand_dims(results, axis=0))
			label_out = np.concatenate(label_out, axis=0)
		else:
			label_out = map_coordinates(label_in, indices, order=1).reshape(shape)
			# label_out = []
			# for sub_vol in label_in:
			# 	results = []
			# 	for input in sub_vol:
			# 		results.append(np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0))
			# 	label_out.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
			# label_out = np.concatenate(label_out, axis=0)
		
		image_out = self._shave(image_out, [self.shave, self.shave])
		label_out = self._shave(label_out, [self.shave, self.shave])
		
		return image_out, label_out


class Valid(Dataset):
	def __init__(self, cfg):
		super(Valid, self).__init__()
		self.folder_name = cfg.DATA.folder_name
		self.data_list = cfg.DATA.valid_data_list

		self.raw = []
		self.labels = []
		f = open(os.path.join(self.folder_name, self.data_list), 'r')
		self.f_list = [x[:-1] for x in f.readlines()]
		self.num = len(self.f_list)
	
	def gen(self, ids):
		files = self.f_list[ids]
		name = files.split(' ')
		raw_name = name[0]
		label_name = name[1]
		raw = np.asarray(cv2.imread(os.path.join(self.folder_name, raw_name), cv2.IMREAD_GRAYSCALE))
		label = np.asarray(cv2.imread(os.path.join(self.folder_name, label_name), cv2.IMREAD_GRAYSCALE))
		h, w = label.shape
		if h == 9959 or h == 9958:
			raw_ = np.zeros((10240, 10240), dtype=np.uint8)
			label_ = np.zeros((10240, 10240), dtype=np.uint8)
			raw_[141:141+h, 141:141+w] = raw
			label_[141:141+h, 141:141+w] = label
			raw = raw_
			label = label_
			del raw_, label_
		raw = raw.astype(np.float32) / 255.0
		label = label // 255
		return raw, label


class Provider(object):
	def __init__(self, stage, cfg):
			#patch_size, batch_size, num_workers, is_cuda=True):
		self.stage = stage
		if self.stage == 'train':
			self.data = Train(cfg)
			self.batch_size = cfg.TRAIN.batch_size
			self.num_workers = cfg.TRAIN.num_workers
		elif self.stage == 'valid':
			# return valid(folder_name, kwargs['data_list'])
			self.data = Valid(cfg)
		else:
			raise AttributeError('Stage must be train/valid')
		self.is_cuda = cfg.TRAIN.is_cuda
		self.data_iter = None
		self.iteration = 0
		self.epoch = 1
	
	def __len__(self):
		return self.data.num_per_epoch
	
	def build(self):
		if self.stage == 'train':
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=False, drop_last=False, pin_memory=True))
		else:
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))
	
	def next(self):
		if self.data_iter is None:
			self.build()
		try:
			batch = self.data_iter.next()
			self.iteration += 1
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
			return batch[0], batch[1]
		except StopIteration:
			self.epoch += 1
			self.build()
			self.iteration += 1
			batch = self.data_iter.next()
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
			return batch[0], batch[1]


if __name__ == '__main__':
	import yaml
	from attrdict import AttrDict
	""""""

	cfg_file = 'complex_unetpp_2_512_padding0.yaml'
	with open('./config/' + cfg_file, 'r') as f:
		cfg = AttrDict( yaml.load(f) )
	
	data = Train(cfg)
	t = time.time()
	for i in range(0, 100):
		im, lb = iter(data).__next__()
		im = (im * 255).astype(np.uint8)
		lb = (lb * 255).astype(np.uint8)
		lb = np.squeeze(lb)
		tmp = np.concatenate([im, lb], axis=1)
		Image.fromarray(tmp).save('../../data/temp/' + str(i).zfill(4)+'.png')
	print(time.time() - t)
	# data = Valid(cfg)
	# for i in range(0, 2):
	# 	raw, label = data.gen(i)
	# 	raw = (raw * 255).astype(np.uint8)
	# 	label = (label * 255).astype(np.uint8)
	# 	tmp = np.concatenate([raw, label], axis=1)
	# 	Image.fromarray(tmp).save('../../data/temp/' + str(i).zfill(4)+'.png')
	# 66.48s