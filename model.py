import os
import numpy as np
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input


baseModel = InceptionV3(weights = 'imagenet', include_top = False)
print('Successfully loaded the InceptionV3 model!\n')

x = baseModel.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation = 'relu')(x)

x = Dropout(0.5)(x)


predictions = Dense(2, activation = 'softmax')(x)


myModel = Model(inputs = baseModel.input, outputs = predictions)


dataGenArgs = dict(preprocessing_function = preprocess_input,
                  rotation_range = 30,
                  width_shift_range = 0.2,
                  height_shift_range = 0.2, 
                  shear_range = 0.2,
                  zoom_range = 0.2,
                  horizontal_flip = True,
                  vertical_flip = True)

trainDataGen = image.ImageDataGenerator(**dataGenArgs)
validDataGen = image.ImageDataGenerator(**dataGenArgs)

trainGenerator = trainDataGen.flow_from_directory(r'{}'.format('trainData'),
                                                 target_size = (299, 299),
                                                 color_mode = 'rgb',
                                                 batch_size = 64,
                                                 class_mode = 'binary',
                                                 shuffle = True)

validGenerator = validDataGen.flow_from_directory(r'{}'.format('validationData'),
                                                 target_size = (299, 299),
                                                 color_mode = 'rgb',
                                                 batch_size = 64,
                                                 class_mode = 'binary',
                                                 shuffle = True)


myModel.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', 
                 metrics = ['accuracy'])



stepSize = trainGenerator.n // trainGenerator.batch_size
validSteps = validGenerator.n // validGenerator.batch_size


myModel.fit_generator(generator = trainGenerator, steps_per_epoch = stepSize, epochs = 20, 
                       validation_data = validGenerator, validation_steps = validSteps)                       
print('\nSucessfully trained the model!')


myModel.save('model.h5')