import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Set parameters
max_features = 10000  # Consider only the top 10,000 most common words
maxlen = 200  # Cut reviews after 200 words
batch_size = 32
embedding_dims = 50
epochs = 5

# Load the data
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to a fixed length
print('Pad sequences (samples x time)')
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Define the model
print('Building model...')
model = Sequential()
model.add(Embedding(max_features, embedding_dims))  # Removed input_length argument
model.add(LSTM(64))  # LSTM layer with 64 units
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print('Training...')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the model
print('Evaluation...')
loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
model.save('model.h5')