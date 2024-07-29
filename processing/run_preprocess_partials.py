import matplotlib.pyplot as plt

import partial_eclipse_tracker

def main():
  tracker = partial_eclipse_tracker.PartialEclipseTracker()
  tracker.preprocess_images()
  tracker.save_preprocess_to_file('preprocess_partials')

  new_tracker = partial_eclipse_tracker.PartialEclipseTracker()
  new_tracker.load_preprocess_from_file('preprocess_partials')

if __name__ == '__main__':
  main()
