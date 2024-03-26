from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

if len(sys.argv) != 2:
    print("Usage: python train.py <number_of_epochs>")
    sys.exit(1) 

epochs = int(sys.argv[1])

# Define the CNN model
classifier = Sequential()

# Add convolutional layers with ReLU activation and max pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output and add fully connected layers
classifier.add(Flatten())
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation='sigmoid'))

# Compile the model
classifier.compile(optimizer=RMSprop(),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and test data
training_set = train_datagen.flow_from_directory('train',
                                                 target_size=(64, 64), 
                                                 batch_size=32, 
                                                 class_mode='binary' 
                                                 )

# different classification give 
# 1. binary - use when you have binary classification problem like cat or dog
# 2. categorical - use when you have more than 2 classes like cat, dog, elephant
# 3. sparse 
# 4. input



# test_set = test_datagen.flow_from_directory('test',
#                                             target_size=(64, 64),
#                                             batch_size=32,
#                                             class_mode='binary')

# Train the model
classifier.fit(training_set,
               steps_per_epoch=len(training_set),
               epochs=epochs)

# Save the trained model
classifier.save('model1.h5')
print("Model saved to disk")
# print the training set length
print(f'Training set length: {len(training_set)}')
