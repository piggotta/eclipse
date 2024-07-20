import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import exifread
import rawpy

import image_loader

PHOTO_PATH = 'photos'

def print_image_params(index: int) -> image_loader.RawImage:
  filename = f'IMG_{index:04d}.CR2'
  filepath = os.path.join(PHOTO_PATH, filename)
  image = image_loader.load_image(filepath)

  print(filename)
  print('  Index:', image.index)
  print('  Time:', image.time)
  print('  Exposure (s):', image.exposure_s)
  print('  F number:', image.f_number)
  print('  ISO:', image.iso)
  print(f'  Raw image: {image.raw_image.shape[0]:d} x '
        f'{image.raw_image.shape[1]:d} pixels')
  print(f'  Raw image: {image.bw_image.shape[0]:d} x '
        f'{image.bw_image.shape[1]:d} pixels')
  print()

  return image

def main():
  start = time.time()
  images = []
  for index in [1001] + list(range(1360, 1370)):
    images.append(print_image_params(index))
  print(f'Time elapsed: {time.time() - start:.3f} s')

  for image in images:
    plt.figure()
    plt.imshow(image.bw_image, cmap='gray')
    plt.colorbar()
    plt.title(f'Image {image.index}')

  plt.show()

if __name__ == '__main__':
  main()
