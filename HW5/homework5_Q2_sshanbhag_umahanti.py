import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from skimage.transform import resize

def data_generator(X, y, batch_size):
    while True:
        idx = np.random.permutation(len(X))
        num_batches = len(X) // batch_size
        for batch_num in range(num_batches):
            start = batch_num * batch_size
            end = (batch_num + 1) * batch_size
            batch_indices = idx[start:end]
            X_batch = np.array([resize(img, (224, 224)) for img in X[batch_indices]])
            y_batch = y[batch_indices]
            yield X_batch, y_batch

# Load the data
faces = np.load("faces.npy")
ages = np.load("ages.npy")

# Convert grayscale images to 3 channels
faces = np.repeat(faces[..., np.newaxis], 3, -1)

# Split the data into training+validation (80%) and testing (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(faces, ages, test_size=0.2, random_state=42)

# Further split the training+validation set into training (64%) and validation (16%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Load the pre-trained ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='linear')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Perform supervised pre-training
for layer in base_model.layers:
    layer.trainable = False

# Fine-tune the last few layers
for layer in model.layers[-4:]:
    layer.trainable = True

# Compile the model
model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')

batch_size = 16
train_generator = data_generator(X_train, y_train, batch_size)
val_generator = data_generator(X_val, y_val, batch_size)

steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_val) // batch_size

# Train the model
history = model.fit(train_generator, validation_data=val_generator, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=10)

# Resize to 224*224
X_train_resized = np.array([resize(img, (224, 224)) for img in X_train])
X_val_resized = np.array([resize(img, (224, 224)) for img in X_val])
X_test_resized = np.array([resize(img, (224, 224)) for img in X_test])

# Evaluate the model on the training, validation, and test sets
mse_train = model.evaluate(X_train_resized, y_train)
mse_val = model.evaluate(X_val_resized, y_val)
mse_test = model.evaluate(X_test_resized, y_test)

# Calculate RMSE
rmse_train = np.sqrt(mse_train)
rmse_val = np.sqrt(mse_val)
rmse_test = np.sqrt(mse_test)

# Print RMSE
print("Train RMSE: {:.2f}".format(rmse_train))
print("Validation RMSE: {:.2f}".format(rmse_val))
print("Test RMSE: {:.2f}".format(rmse_test))