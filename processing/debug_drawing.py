import time

import numpy as np
import matplotlib.pyplot as plt

import drawing

def main():
  num_repeats = 10
  image_shape = (500, 500)
  center = (200, 400)
  radius = 130

  for renderer in ['float_raster']:
    start = time.time()
    for ind in range(num_repeats):
      image = drawing.draw_circle(image_shape, center, radius)
    elapsed_per_render = (time.time() - start) / num_repeats
    print(f'{renderer}: {elapsed_per_render / 1e-3:.3f} ms per image')

    plt.figure()
    plt.imshow(image)
    plt.title(renderer)

  plt.show()

if __name__ == '__main__':
  main()
