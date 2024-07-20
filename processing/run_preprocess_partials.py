import matplotlib.pyplot as plt

import partial_eclipse_tracker

if __name__ == '__main__':
  tracker = partial_eclipse_tracker.PartialEclipseTracker()
  #tracker.preprocess_images(max_images=30, start_ind=1300)
  tracker.preprocess_images(max_images=3)
  tracker.save_preprocess_to_file('dummy')

  new_tracker = partial_eclipse_tracker.PartialEclipseTracker()
  new_tracker.load_preprocess_from_file('dummy')

  plt.show()
