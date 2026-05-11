import tensorflow as tf

from preprocessing import train_generator
from preprocessing import val_generator

from models import build_parallel_model

EPOCHS = 5

model = build_parallel_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

model.save('../models/parallel_model.h5')

print("Model Saved Successfully")
