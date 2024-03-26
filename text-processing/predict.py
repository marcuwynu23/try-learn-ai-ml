import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the trained model
model = load_model('model.h5')

# New text data for prediction
# Create a tokenizer object
tokenizer = Tokenizer()

new_texts = [
	"This movie is fantastic! I loved every moment of it.",
	"The acting was terrible and the plot was boring."
]

# Tokenize and pad the new text data
maxlen = 200  # Same as the maxlen used during training
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_sequences = pad_sequences(new_sequences, maxlen=maxlen)

# Make predictions
predictions = model.predict(new_sequences)

# Convert predictions to binary labels
binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]

# Print predictions
for text, pred in zip(new_texts, binary_predictions):
    print(f"Text: {text}")
    print(f"Predicted Label: {'Positive' if pred == 1 else 'Negative'}")
    print()
