import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Define the freeze function, outptut freezed graph

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

    """

    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by

    constants taking their current value in the session. The new graph will be

    pruned so subgraphs that are not necessary to compute the requested

    outputs are removed.

    @param session The TensorFlow session to be frozen.

    @param keep_var_names A list of variable names that should not be frozen,

                          or None to freeze all the variables in the graph.

    @param output_names Names of the relevant graph outputs.

    @param clear_devices Remove the device directives from the graph for better portability.

    @return The frozen graph definition.

    """

    graph = session.graph

    with graph.as_default():

        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []

        output_names += [v.op.name for v in tf.global_variables()]

        input_graph_def = graph.as_graph_def()

        if clear_devices:

            for node in input_graph_def.node:

                node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(

            session, input_graph_def, output_names, freeze_var_names)

    return frozen_graph

    

mnist = tf.keras.datasets.mnist

print(tf.__version__)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train, x_test


#y_train[y_train > 0] = 1
#y_test[y_test > 0] = 1 #to have only two classes 0 or 1 instead of 10 0 => 0 enything else =>1

# Reshape (add color channel)
#x_train = x_train.reshape(-1, 28, 28, 1)
#x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#model = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(28, 28)),
#  tf.keras.layers.Dense(512, activation=tf.nn.relu),
#  tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#])

#model = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(28, 28,1)),
#  tf.keras.layers.Dense(128, activation='relu'),
#  tf.keras.layers.Dense(10, activation='softmax')
#])


#10 classes


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)), 
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

#two classes

#model = tf.keras.models.Sequential([
#  tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)),
#  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #tf.keras.layers.Dropout(0.25),
#  tf.keras.layers.Flatten(),
#  tf.keras.layers.Dense(128, activation='relu'),
  #tf.keras.layers.Dropout(0.5),
#  tf.keras.layers.Dense(2, activation='softmax')
#])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=25)
model.evaluate(x_test, y_test)

# Save tf.keras model in HDF5 format.
keras_file = "first_model_mcu.h5"
tf.keras.models.save_model(model, keras_file)

graph_path = os.path.join(".", "con1d_model.png")
plot_model(model, to_file=graph_path, show_shapes=True)  



# Convert to TensorFlow Lite model.
#converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)

# Convert the model to the TensorFlow Lite format without quantization
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()

# Save the model to disk
#open("model.tflite", "wb").write(tflite_model)

# Convert the model to the TensorFlow Lite format with quantization
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.post_training_quantize=True
#tflite_model = converter.convert()

# Save the model to disk
#open("model_quantized.tflite", "wb").write(tflite_model)

#basic_model_size = os.path.getsize("model.tflite")
#print("Basic model is %d bytes" % basic_model_size)
#quantized_model_size = os.path.getsize("model_quantized.tflite")
#print("Quantized model is %d bytes" % quantized_model_size)
#difference = basic_model_size - quantized_model_size
#print("Difference is %d bytes" % difference)