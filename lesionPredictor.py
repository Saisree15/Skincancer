
import os
import glob
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input

print('Loading the model...\n')


myModel = load_model('model.h5')

print('\nSuccessfully loaded the model!')


class Session: 

  def __init__ (self, modelName):
    self.modelName = modelName

  def imagePredict (self, imagePath):   
    testImage = load_img (imagePath, target_size = (299, 299))


    x = img_to_array (testImage)

    x = np.expand_dims (x, axis = 0)
    x = preprocess_input (x)


    digitPrediction = np.argmax (self.modelName.predict(x))

    if digitPrediction == 0:
      return 'Non-Melanoma'

    else:
      return 'Melanoma'

  def showPrediction (self, imagePath):

    lesionPrediction = self.imagePredict('{}'.format(imagePath))
    
    print('The model predicts that this lesion is: {}'.format(lesionPrediction))


imagePath = input('\nEnter the image file path: ')


while not os.path.exists ('{}'.format(imagePath)):
    print('Error! That path does not exist. Try again.')
    print('Note that quotations should NOT be included in the path.')

    imagePath = input('\nEnter the image file path: ')
    print('')


currentSession = Session(myModel)
currentSession.imagePredict(r'{}'.format(imagePath))
currentSession.showPrediction(r'{}'.format(imagePath))