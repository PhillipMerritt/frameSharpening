import cv2
import numpy as np
from keras import models


model = models.load_model("data/saved_models/encoder1_2.h5")
patch_size = (100, 100)
y_stride = 25
x_stride = 25
batch_size = 16 # number of images to get patches from
patches_per_batch = 32 # total number of patches per batch of images
max_patches = patches_per_batch // batch_size   # number of patches to get from each image in the batch

# takes the filename of an uploaded video in the data folder
# and return the name of the sharpened version also in the data folder
def sharpen_video(filename):
    print("Getting frames")
    frames = extract_frames('data/' + filename)
    frames = np.array(frames) / 255. # convert to np array and normalize rgb values to 0-1

    print("Sharpening each frame")
    print("ETA: {} minutes".format(frames.shape[0] * 4.0 / 60.0))
    sharp_frames = [sharpen_image(model, img) for img in frames]

    print("Creating new video")
    video_name = "sharpened_"+filename
    height, width = sharp_frames.shape[1:3]
    video = cv2.VideoWriter("data/" + video_name, 0, 1, (width,height))

    for img in sharp_frames:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    return video_name

def bgr_to_rgb(img):
  def helper(bgr):
    b, g, r = bgr
    return [r, g, b]

  for y in range(100):
    for x in range(100):
      img[y][x] = helper(img[y][x])

def extract_frames(filename):
  vidcap = cv2.VideoCapture(filename)
  success, image = vidcap.read()

  frames = []

  while success:
    bgr_to_rgb(image)
    frames.append(image)
    success, image = vidcap.read()
  
  return frames

def reconstruct_image(patches, ranges, weights):
  img = np.zeros(weights.shape)
  for i, (y0, y1, x0, x1) in enumerate(ranges):
    img[y0:y1,x0:x1] += patches[i]
  
  return np.array(np.clip((img / weights) * 255, 0.0, 255.0), dtype=np.uint8)

def sharpen_image(model, img):
  patches = []
  weights = np.zeros(img.shape)
  ranges = []
  for y in range(patch_size[1], img.shape[0] + 1 + (y_stride - (img.shape[0] % y_stride)), y_stride):
    if y > img.shape[0]:
      y = img.shape[0]

    for x in range(patch_size[0], img.shape[1] + 1 + (x_stride - (img.shape[1] % x_stride)), x_stride):
      if x > img.shape[1]:
        x = img.shape[1]

      y0 = y - patch_size[1]
      x0 = x - patch_size[0]
      patches.append(img[y0 : y, x0 : x])
      weights[y0 : y, x0 : x] += 1
      ranges.append([y0, y, x0, x])

  patches = np.array(patches)
  patch_batch_count = batch_size * max_patches
  preds = []
  for i in range(0, patches.shape[0], patch_batch_count):
    preds.append(np.array(model.predict(patches[i : i + patch_batch_count])))

  preds = np.concatenate(preds)

  return reconstruct_image(preds, ranges, weights)