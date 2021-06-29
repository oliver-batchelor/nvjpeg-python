from turbojpeg import TurboJPEG
from nvjpeg import NvJpeg

import cv2
import time


from multiprocessing import Process
import multiprocessing

from threading import Thread
from queue import Queue

import argparse

class Timer:    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


class CvJpeg(object):
  def encode(self, image):
    result, compressed = cv2.imencode('.jpg', image)



class Threaded(object):
  def __init__(self, create_jpeg, size=8):
        # Image file writers
    self.queue = Queue(size)
    self.threads = [Thread(target=self.encode_thread, args=()) 
        for _ in range(size)]

    self.jpeg = create_jpeg()

    
    for t in self.threads:
        t.start()


  def encode_thread(self):

    item = self.queue.get()
    while item is not None:
        result = self.jpeg.encode(item)
        item = self.queue.get()


  def encode(self, image):
    self.queue.put(image)


  def stop(self):
      for _ in self.threads:
          self.queue.put(None)

      for t in self.threads:
        t.join()
      


def bench_threaded(create_encoder, images, threads):
  threads = Threaded(create_encoder, threads)

  with Timer() as t:
    for image in images:
      threads.encode(image)

    threads.stop()

  return len(images) / t.interval




def bench_encoder(create_encoder, images):
  encoder = create_encoder()

  with Timer() as t:
    for image in images:
      encoder.encode(image)

  return len(images) / t.interval


def main(args):
  image = cv2.imread(args.filename, cv2.IMREAD_COLOR)

  images = [image] * args.n
  num_threads = args.j

  if num_threads > 1:
    # print(f'turbojpeg threaded j={num_threads}: {bench_threaded(TurboJPEG, images, num_threads):>5.1f} images/s')
    print(f'nvjpeg threaded j={num_threads}: {bench_threaded(NvJpeg, images, num_threads):>5.1f} images/s')
  # print(f'opencv threaded j={num_threads}: {bench_threaded(CvJpeg, images, num_threads):>5.1f} images/s')
  
  else:
    # print(f'turbojpeg: {bench_encoder(TurboJPEG, images):>5.1f} images/s')
    print(f'nvjpeg: {bench_encoder(NvJpeg, images):>5.1f} images/s')
    # print(f'opencv: {bench_encoder(CvJpeg, images):>5.1f} images/s')



if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Jpeg encoding benchmark.')
  parser.add_argument('filename', type=str, help='filename of image to use')

  parser.add_argument('--j', default=1, type=int, help='run multi-threaded')
  parser.add_argument('--n', default=100, type=int, help='number of images to encode')

  args = parser.parse_args()
  main(args)
