from nvjpeg import NvJpeg
from os import path
import numpy as np

import cv2

import argparse



if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Jpeg encoding benchmark.')
  parser.add_argument('filename', type=str, help='filename of image to use')

  jpeg = NvJpeg()

  args = parser.parse_args()
  image = cv2.imread(args.filename, cv2.IMREAD_COLOR)

  data = jpeg.encode(image)
  # decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)


  # cv2.imshow("image", decoded)
  # cv2.waitKey()

  # print(np.array(data).sum())

  with open(path.join("out", path.basename(args.filename)), "wb") as f:
    f.write(data)
