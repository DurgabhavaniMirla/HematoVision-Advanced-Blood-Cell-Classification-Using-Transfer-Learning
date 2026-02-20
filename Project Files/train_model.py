import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. SETUP
TRAIN_DIR = r"C:\Users\durga\OneDrive\Desktop\Hematovision\bloodcells dataset\dataset2-master\dataset2-master\images\TRAIN"
MODEL_PATH = "blood_cell.h5"
IMG_SIZE = 224
BATCH_SIZE = 32

# 2. AUGMENTATION (Heavy)
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

train_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
                                        batch_size=BATCH_SIZE, class_mode="categorical",
                                        subset="training", shuffle=True)

val_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
                                      batch_size=BATCH_SIZE, class_mode="categorical",
                                      subset="validation", shuffle=False)

with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# 3. BUILD MODEL
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False 

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

# 4. STAGE 1: WARM UP
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_gen, validation_data=val_gen, epochs=10)

# 5. STAGE 2: FINE TUNING (The "Secret Sauce")
print("Starting Fine-Tuning...")
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Tiny learning rate
              loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy')]

model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

print("✅ FINAL MODEL SAVED!")