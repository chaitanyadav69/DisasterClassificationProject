import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from preprocessing import test_generator

model = tf.keras.models.load_model(
    '../models/parallel_model.h5'
)

predictions = model.predict(test_generator)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

print(classification_report(
    true_classes,
    predicted_classes
))

print(confusion_matrix(
    true_classes,
    predicted_classes
))
