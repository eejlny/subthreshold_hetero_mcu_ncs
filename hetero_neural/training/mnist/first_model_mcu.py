import tensorflow as tf
import flatbuffer_2_tfl_micro as save_tflm

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


# set learning phase for no training
tf.keras.backend.set_learning_phase(0)

# load weights & architecture into new model
loaded_model = tf.keras.models.load_model('first_model_mcu.h5')
# Now we can write out the TensorFlow compatible checkpoint and inference graph which will be used later with the freeze_graph.py script to create the frozen model:
# make list of output node names
output_names=[out.op.name for out in loaded_model.outputs]
input_names=[out.op.name for out in loaded_model.inputs]

loaded_model.summary()

print('input  node is{}'.format(input_names))

# make list of output and input node names

print('output node is{}'.format(output_names))

#converter = tf.lite.TFLiteConverter.from_keras_model_file('first_model_mcu.h5')
# Set the optimization flag.
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Enforce full-int8 quantization (except inputs/outputs which are always float)
# So to ensure that the converted model is fully quantized 
# (make the converter throw an error if it encounters an operation it cannot quantize), 
# and to use integers for the model's input and output, 
# you need to convert the model again using these additional configurations:
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8  # or tf.uint8
#converter.inference_output_type = tf.int8  # or tf.uint8
# Provide a representative dataset to ensure we quantize correctly.
#converter.representative_dataset = representative_dataset
#tflite_model = converter.convert()
#open('./LeNET-MNIST_int8ops.tflite', 'wb').write(tflite_model)

# Freeze graph

frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=output_names)

# Store graph to pb(Protocol Buffers) file

tf.train.write_graph(frozen_graph, "./", "frozen_graph_first_model_mcu.pb", as_text=False)