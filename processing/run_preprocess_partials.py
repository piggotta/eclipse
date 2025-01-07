import matplotlib.pyplot as plt

import partial_eclipse_tracker
import util

def main():
  util.print_title('Preprocess partial eclipse images')

  preprocessor = partial_eclipse_tracker.PartialEclipsePreprocessor()
  preprocessor.preprocess_images()
  preprocessor.save_to_file('preprocess_partials')

  new_preprocessor = partial_eclipse_tracker.PartialEclipsePreprocessor()
  new_preprocessor.load_from_file('preprocess_partials')

if __name__ == '__main__':
  main()
