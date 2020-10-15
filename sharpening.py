import cv2
import numpy as np
from keras import models
from tqdm import tqdm
import imageio
import PIL.Image


model_2x = models.load_model("data/saved_models/production_autoencoder_2x.h5")
model_4x = models.load_model("data/saved_models/production_autoencoder_4x.h5")
patch_size = (100, 100)
y_stride = 25
x_stride = 25
patches_per_batch = 16 # total number of patches per batch of images

video_extensions = {'mp4', 'mpeg-4', 'mov', 'avi'}

# takes the filename of an uploaded video or image in the data folder
# and return the name of the sharpened version also in the data folder
def sharpen_upload(filename):
  # check if the file is a video
  if filename.rsplit('.', 1)[1].lower() in video_extensions:
    print("Getting frames")
    frames, fps = extract_frames('data/' + filename)
    frames = np.array(frames) / 255. # convert to np array and normalize rgb values to 0-1

    print("Sharpening each frame")
    print("ETA: {} minutes".format(frames.shape[0] * 4.0 / 60.0))
    sharp_frames = [sharpen_image(model_2x, img) for img in tqdm(frames)]

    print("Creating new video")
    video_name = "sharpened_"+filename
    height, width = sharp_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter("data/" + video_name, fourcc, fps, (width,height))

    for img in tqdm(sharp_frames):
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    return video_name
  else: # else it's an image
    og_img = PIL.Image.open('data/' + filename)
    og_img = np.array(og_img.convert("RGB"))
    img = og_img / 255.
    #img = pixalate_image(img)
    img = sharpen_image(model_2x, img)

    og_img = np.array(og_img, dtype=np.uint8)

    img_name = 'sharpened_' + filename
    imageio.imwrite('data/' + img_name, img)

    return img_name

def bgr_to_rgb(img):
  def helper(bgr):
    b, g, r = bgr
    return [r, g, b]

  for y in range(100):
    for x in range(100):
      img[y][x] = helper(img[y][x])

def extract_frames(filename):
  vidcap = cv2.VideoCapture(filename)
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  success, image = vidcap.read()
  dim = (image.shape[1] * 2, image.shape[0] * 2)
  frames = []

  while success:
    bgr_to_rgb(image)
    image = cv2.resize(image, dim)
    frames.append(image)
    success, image = vidcap.read()
  
  return frames, fps

def reconstruct_image(patches, ranges, weights):
  img = np.zeros(weights.shape)
  for i, (y0, y1, x0, x1) in enumerate(ranges):
    img[y0:y1,x0:x1] += patches[i]
  
  return np.array(np.clip((img / weights) * 255, 0.0, 255.0), dtype=np.uint8)


def pixalate_image(image, scale_percent = 50):
  og_w = image.shape[1]
  og_h = image.shape[0]
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dim = (width, height)

  small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
  # scale back to original size
  width = og_w
  height = og_h
  dim = (width, height)

  low_res_image = cv2.resize(small_image, dim, interpolation = cv2.INTER_AREA)

  return low_res_image

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
  preds = []
  for i in range(0, patches.shape[0], patches_per_batch):
    preds.append(np.array(model.predict(patches[i : i + patches_per_batch])))

  preds = np.concatenate(preds)

  return reconstruct_image(preds, ranges, weights)