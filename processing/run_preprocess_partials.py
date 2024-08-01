import matplotlib.pyplot as plt

import partial_eclipse_tracker

def main():
  options = partial_eclipse_tracker.PartialEclipseTrackerOptions()
  tracker = partial_eclipse_tracker.PartialEclipseTracker(options)
  tracker.preprocess_images()
  tracker.save_preprocess_to_file('preprocess_partials')

  new_tracker = partial_eclipse_tracker.PartialEclipseTracker(options)
  new_tracker.load_preprocess_from_file('preprocess_partials')

if __name__ == '__main__':
  main()
