import camera_utils

def main():
  print('Available cameras: ')
  for name in camera_utils.available_camera_names():
    print(' ', name)

if __name__ == '__main__':
  main()

