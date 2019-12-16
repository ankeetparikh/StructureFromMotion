import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD




IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172


def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  nb_train_samples = get_nb_files(args.train_dir)
  nb_classes = len(glob.glob(args.train_dir + "/*"))
  nb_val_samples = get_nb_files(args.val_dir)
  nb_epoch = int(args.nb_epoch)
  batch_size = int(args.batch_size)

  # data prep
  train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )
  test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

  train_generator = train_datagen.flow_from_directory(
    args.train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  validation_generator = test_datagen.flow_from_directory(
    args.val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  # setup model
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, nb_classes)

  # transfer learning
  # setup_to_transfer_learn(model, base_model)

  # history_tl = model.fit_generator(
  #       train_generator,
  #       steps_per_epoch= 1,
  #       epochs=1,
  #       validation_data=validation_generator,
  #       validation_steps= 3)

  # fine-tuning
  setup_to_finetune(model)

  history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=3
    )

  model.save(args.output_model_file)

  if args.plot:
    plot_training(history_ft)


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')
  plt.show()

def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]

def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  labels = ("cat", "dog")
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()

def Part2a():
  from keras.preprocessing import image as image_utils
  from imagenet_utils import decode_predictions
  from imagenet_utils import preprocess_input
  from resnet50 import ResNet50
  import numpy as np
  import argparse
  import cv2

  import numpy as np
  import json

  from keras.utils.data_utils import get_file
  from keras import backend as K
  imgloc = "Images/partaimages/lynx1.jpg"
  # load the original image via OpenCV so we can draw on it and display
  # it to our screen later
  orig = cv2.imread(imgloc)

  # load the input image using the Keras helper utility while ensuring
  # that the image is resized to 224x224 pxiels, the required input
  # dimensions for the network -- then convert the PIL image to a
  # NumPy array
  print("[INFO] loading and preprocessing image...")
  image = image_utils.load_img(imgloc, target_size=(224, 224))
  image = image_utils.img_to_array(image)

  # our image is now represented by a NumPy array of shape (3, 224, 224),
  # but we need to expand the dimensions to be (1, 3, 224, 224) so we can
  # pass it through the network -- we'll also preprocess the image by
  # subtracting the mean RGB pixel intensity from the ImageNet dataset
  image = np.expand_dims(image, axis=0)
  image = preprocess_input(image)

  # load the ResNet50 network
  print("[INFO] loading network...")
  model = ResNet50(weights="imagenet")
   
  # classify the image
  print("[INFO] classifying image...")
  preds = model.predict(image)
  P = decode_predictions(preds)
  (imagenetID, label, prob) = P[0][0]

  # display the predictions to our screen
  print("ImageNet ID: {}, Label: {}".format(imagenetID, label))
  cv2.putText(orig, "Label: {}".format(label), (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
  cv2.imshow("Classification", orig)
  cv2.waitKey(0)


def main():
  # uncomment this line and the next one for part 2a
  # Part2a();

  # uncomment the following for part 2d
  a = argparse.ArgumentParser()
  a.add_argument("--train_dir")
  a.add_argument("--val_dir")
  a.add_argument("--nb_epoch", default=NB_EPOCHS)
  a.add_argument("--batch_size", default=BAT_SIZE)
  a.add_argument("--output_model_file", default="inceptionv3-ft.model")
  a.add_argument("--plot", action="store_true")

  args = a.parse_args()
  if args.train_dir is None or args.val_dir is None:
    a.print_help()
    sys.exit(1)

  if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
    print("directories do not exist")
    sys.exit(1)

  train(args)
  
  model = load_model("inceptionv3-ft.model");
  
  catcat = 0;
  catdog = 0;
  dogdog = 0;
  dogcat = 0;

  for i in xrange(500, 1000):
    img = Image.open("Images/test/cat." + str(i) +".jpg");
    preds = predict(model, img, (IM_HEIGHT, IM_WIDTH))
    if(preds[0] > preds[1]):
      catcat = catcat + 1;
    else:
      catdog = catdog + 1;

  for i in xrange(500, 1000):
    img = Image.open("Images/test/dog." + str(i) +".jpg");
    preds = predict(model, img, (IM_HEIGHT, IM_WIDTH))
    if(preds[1] > preds[0]):
      dogdog = dogdog + 1;
    else:
      dogcat = dogcat + 1;

  print(catcat)
  print(catdog)
  print(dogdog)
  print(dogcat)

main();
