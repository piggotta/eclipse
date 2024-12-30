import matplotlib.pyplot as plt
import numpy as np

import constants
import eclipse_images
import eclipse_tracker
import filepaths
import image_loader
import partial_eclipse_tracker
import raw_processor


def main():
  # Load raw image processor configuration from file.
  processor = raw_processor.RawProcessor()
  processor.load_calibration_from_file('calibration')

  # Load full eclipse image sequence from file
  sequence = eclipse_images.load_sequence('full_sequence')

  # Create renderer object.
  renderer = eclipse_images.Renderer(processor, sequence)

  # Find total eclipse images.
  totals_unix_time_s = [entry.unix_time_s for entry in sequence.image_metadata
                        if entry.image_type == eclipse_images.ImageType.HDR]

  # Load corona fit.
  p_fit = np.load(filepaths.corona_fit())['p_fit']

  # Render images.
  for ind in 1, 10, -2:
    renderer.render_total(totals_unix_time_s[ind], p_fit)

  plt.show()

if __name__ == '__main__':
  main()
