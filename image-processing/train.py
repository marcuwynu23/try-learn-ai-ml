import tensorflow as tf
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python train.py <epochs>")
    sys.exit(1)

epochs = int(sys.argv[1])


# Load images and labels from files
def load_images_and_labels(image_dir, target_shape=(224, 224)):
    images = []
    labels = []
    labelNames = {}

    # Read image files from the image directory
    image_files = os.listdir(image_dir)

    # Keep track of assigned labels
    assigned_labels = set()

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_tensor = tf.image.decode_image(image_data)  # Decode image data
            resized_image = tf.image.resize(image_tensor, target_shape)
            images.append(resized_image)

        # Convert label string to integer
        try:
            image_label = image_file.split(".")[0]
            label = int(image_label)
            if label in assigned_labels:
                raise ValueError(f"Label '{label}' is already assigned to another image.")
            labels.append(label)
            assigned_labels.add(label)
            labelNames[label] = image_label
        except ValueError:
            # Handle case where filename is not a valid integer label
            print(f"Warning: Filename '{image_file}' is not a valid integer label. Assigning a unique integer label.")
            label = len(assigned_labels)  # Assign a unique integer label
            labels.append(label)
            assigned_labels.add(label)
            labelNames[label] = image_label

    return images, labels, labelNames


# Example usage
images, labels,labelNames = load_images_and_labels("images")


# Store labelNames into a file
with open("animal.names", "w") as f:
    for label in labelNames:
        f.write(f"{label} {labelNames[label]}\n")


# Convert lists to tensors
images_tensor = tf.stack(images)
labels_tensor = tf.constant(labels, dtype=tf.int32)

# Shuffle images and labels
shuffle_indices = tf.range(start=0, limit=tf.shape(images_tensor)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(shuffle_indices)
shuffled_images = tf.gather(images_tensor, shuffled_indices)
shuffled_labels = tf.gather(labels_tensor, shuffled_indices)

# Split the shuffled data into training and validation sets
split_index = tf.shape(shuffled_images)[0] // 2
train_images, val_images = shuffled_images[:split_index], shuffled_images[split_index:]
train_labels, val_labels = shuffled_labels[:split_index], shuffled_labels[split_index:]

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),  # Removed input_shape argument
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels))

print("Training complete.")

# Save the model
model.save("model.h5")  