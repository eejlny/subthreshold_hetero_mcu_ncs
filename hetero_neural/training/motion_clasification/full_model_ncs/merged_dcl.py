# -- coding: utf-8 --

import os
#os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

import sys
import tensorflow as tf
from tensorflow.keras import Input, Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Dense, Activation, Concatenate, Flatten
from tensorflow.keras.utils import to_categorical,plot_model
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

lstmlast = 1

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)

from project_utils import create_folder_try
from root_dir import ROOT_DIR


def load_data(data_path):
    train_path = os.path.join(data_path, "train")
    train_X_path = os.path.join(train_path, "signals")

    X_trainS1_x = np.loadtxt(os.path.join(train_X_path, "body_acc_x_train.txt"))
    X_trainS1_y = np.loadtxt(os.path.join(train_X_path, "body_acc_y_train.txt"))
    X_trainS1_z = np.loadtxt(os.path.join(train_X_path, "body_acc_z_train.txt"))
    X_trainS1 = np.array([X_trainS1_x, X_trainS1_y, X_trainS1_z])
    X_trainS1 = X_trainS1.transpose([1, 2, 0])

    X_trainS2_x = np.loadtxt(os.path.join(train_X_path, "body_gyro_x_train.txt"))
    X_trainS2_y = np.loadtxt(os.path.join(train_X_path, "body_gyro_y_train.txt"))
    X_trainS2_z = np.loadtxt(os.path.join(train_X_path, "body_gyro_z_train.txt"))
    X_trainS2 = np.array([X_trainS2_x, X_trainS2_y, X_trainS2_z])
    X_trainS2 = X_trainS2.transpose([1, 2, 0])

    X_trainS3_x = np.loadtxt(os.path.join(train_X_path, "total_acc_x_train.txt"))
    X_trainS3_y = np.loadtxt(os.path.join(train_X_path, "total_acc_y_train.txt"))
    X_trainS3_z = np.loadtxt(os.path.join(train_X_path, "total_acc_z_train.txt"))
    X_trainS3 = np.array([X_trainS3_x, X_trainS3_y, X_trainS3_z])
    X_trainS3 = X_trainS3.transpose([1, 2, 0])

    Y_train = np.loadtxt(os.path.join(train_path, "y_train.txt"))
    Y_train = to_categorical(Y_train - 1.0)  


    test_path = os.path.join(data_path, "test")
    test_X_path = os.path.join(test_path, "signals")

    X_valS1_x = np.loadtxt(os.path.join(test_X_path, "body_acc_x_test.txt"))
    X_valS1_y = np.loadtxt(os.path.join(test_X_path, "body_acc_y_test.txt"))
    X_valS1_z = np.loadtxt(os.path.join(test_X_path, "body_acc_z_test.txt"))
    X_valS1 = np.array([X_valS1_x, X_valS1_y, X_valS1_z])

    X_valS1 = X_valS1.transpose([1, 2, 0])

    #print(X_valS1[0,0:]) #row 0 all elements

    #sys.exit(0)


    X_valS2_x = np.loadtxt(os.path.join(test_X_path, "body_gyro_x_test.txt"))
    X_valS2_y = np.loadtxt(os.path.join(test_X_path, "body_gyro_y_test.txt"))
    X_valS2_z = np.loadtxt(os.path.join(test_X_path, "body_gyro_z_test.txt"))
    X_valS2 = np.array([X_valS2_x, X_valS2_y, X_valS2_z])
    X_valS2 = X_valS2.transpose([1, 2, 0])

    X_valS3_x = np.loadtxt(os.path.join(test_X_path, "total_acc_x_test.txt"))
    X_valS3_y = np.loadtxt(os.path.join(test_X_path, "total_acc_y_test.txt"))
    X_valS3_z = np.loadtxt(os.path.join(test_X_path, "total_acc_z_test.txt"))
    X_valS3 = np.array([X_valS3_x, X_valS3_y, X_valS3_z])
    X_valS3 = X_valS3.transpose([1, 2, 0])

    Y_val = np.loadtxt(os.path.join(test_path, "y_test.txt"))
    Y_val = to_categorical(Y_val - 1.0)


    return X_trainS1, X_trainS2, X_trainS3, Y_train, X_valS1, X_valS2, X_valS3, Y_val


class Metrics(Callback):

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, batch, logs=None):
        val_targ = self.validation_data[-3]

        val_value = [x for x in self.validation_data[0:-3]]
        y_pred = np.asarray(self.model.predict(val_value))

        precision, recall, f_score, _ = precision_recall_fscore_support(
            val_targ, (y_pred > 0.5).astype(int), average='micro')
        #print "— val_f1: % f — val_precision: % f — val_recall % f" % (f_score, precision, recall)


def main(data_path, output_path):
    X_trainS1, X_trainS2, X_trainS3, Y_train, X_valS1, X_valS2, X_valS3, Y_val = load_data(data_path)



    #scaler = MinMaxScaler(feature_range=(-1,1))

    scaler = MinMaxScaler()

    #X_valS1 = scaler.fit_transform(X_valS1.reshape(-1, X_valS1.shape[-1])).reshape(X_valS1.shape)
    #X_valS2 = scaler.fit_transform(X_valS2.reshape(-1, X_valS2.shape[-1])).reshape(X_valS2.shape)
    #X_valS3 = scaler.fit_transform(X_valS3.reshape(-1, X_valS3.shape[-1])).reshape(X_valS3.shape)


    #X_trainS1 = scaler.fit_transform(X_trainS1.reshape(-1, X_trainS1.shape[-1])).reshape(X_trainS1.shape)
    #X_trainS2 = scaler.fit_transform(X_trainS2.reshape(-1, X_trainS2.shape[-1])).reshape(X_trainS2.shape)
    #X_trainS3 = scaler.fit_transform(X_trainS3.reshape(-1, X_trainS3.shape[-1])).reshape(X_trainS3.shape)

    #X_trainS1 = (X_trainS1 * 255).astype(int)
    #X_trainS2 = (X_trainS2 * 255).astype(int)
    #X_trainS3 = (X_trainS3 * 255).astype(int)
    #X_valS1 = (X_valS1 * 255).astype(int)
    #X_valS2 = (X_valS2 * 255).astype(int)
    #X_valS3 = (X_valS3 * 255).astype(int)

    #X_trainS1 = X_trainS1/255
    #X_trainS2 = X_trainS2/255
    #X_trainS3 = X_trainS3/255
    #X_valS1 = X_valS1/255
    #X_valS2 = X_valS2/255
    #X_valS3 = X_valS3/255

    #select sample 100 row, all the 128 column and a particular depth value 

    savetxt('./X_trainS1_q_x.csv', X_trainS1[0,:,0], newline=',',delimiter=',',fmt='%d')

    #savetxt('./X_trainS1_q_x.csv', X_trainS1[0,:,0], newline=',',delimiter=',',fmt='%1.3f')


    savetxt('./X_trainS1_q_y.csv', X_trainS1[0,:,1], newline=',',delimiter=',',fmt='%d')

    savetxt('./X_trainS1_q_z.csv', X_trainS1[0,:,2], newline=',',delimiter=',',fmt='%d')

    savetxt('./X_trainS2_q_x.csv', X_trainS2[0,:,0], newline=',',delimiter=',',fmt='%d')

    savetxt('./X_trainS2_q_y.csv', X_trainS2[0,:,1], newline=',',delimiter=',',fmt='%d')

    savetxt('./X_trainS2_q_z.csv', X_trainS2[0,:,2], newline=',',delimiter=',',fmt='%d')

    savetxt('./X_trainS3_q_x.csv', X_trainS3[0,:,0], newline=',',delimiter=',',fmt='%d')

    savetxt('./X_trainS3_q_y.csv', X_trainS3[0,:,1], newline=',',delimiter=',',fmt='%d')

    savetxt('./X_trainS3_q_z.csv', X_trainS3[0,:,2], newline=',',delimiter=',',fmt='%d')

    #print(X_trainS1[100,:,0])
    #print(Y_train[0])
    #exit()

    #a = np.array([[0, 1, 6],
    #              [2, 4, 1]])

    print("Max value in train data",np.max(X_trainS1))

    print("Min value in train data",np.min(X_trainS1))

    #exit()




    epochs = 60
    #epochs = 5
    batch_size = 256
    kernel_size = 3
    pool_size = 2
    dropout_rate = 0.15
    n_classes = 6

    f_act = 'relu'


    main_input1 = Input(shape=(128, 3), name='main_input1')
    main_input2 = Input(shape=(128, 3), name='main_input2')
    main_input3 = Input(shape=(128, 3), name='main_input3')

    def cnn_lstm_cell(main_input):
        sub_model = Conv1D(512, kernel_size, input_shape=(128, 3), activation=f_act, padding='same')(main_input)
        sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        #sub_model = Dropout(dropout_rate)(sub_model)
        sub_model = Conv1D(64, kernel_size, activation=f_act, padding='same')(sub_model)
        sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        #sub_model = Dropout(dropout_rate)(sub_model)
        sub_model = Conv1D(32, kernel_size, activation=f_act, padding='same')(sub_model)
        #sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        #sub_model = LSTM(128, return_sequences=True)(sub_model)
        #sub_model = LSTM(128, return_sequences=True)(sub_model)
        sub_model = LSTM(128)(sub_model)
        #main_output = Dropout(dropout_rate)(sub_model)
        main_output = sub_model
        return main_output

    def buildLstmLayer():
        return tf.keras.layers.StackedRNNCells([
            tf.lite.experimental.nn.TFLiteLSTMCell(10, use_peepholes=True, forget_bias=0, name="rnn1"),
            tf.lite.experimental.nn.TFLiteLSTMCell(10, num_proj=8, forget_bias=0, name="rnn2")
        ])

    def cnn_lstm_cell_tflite(main_input):
       
        #sub_model = Conv1D(512, kernel_size, input_shape=(128, 3,1), activation=f_act, padding='same')(main_input)
        sub_model = Conv1D(512, kernel_size, input_shape=(128, 3), activation='relu', padding='same')(main_input)
        ##sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        ##sub_model = Dropout(dropout_rate)(sub_model)
        sub_model = Conv1D(64, kernel_size, activation=f_act, padding='same')(sub_model)
        ##sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        ##sub_model = Dropout(dropout_rate)(sub_model)
        sub_model = Conv1D(32, kernel_size, activation=f_act, padding='same')(sub_model)
        ##sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        #sub_model = LSTM(128, return_sequences=True)(sub_model)
        #sub_model = LSTM(128, return_sequences=True)(sub_model)
        #sub_model = LSTM(128)(sub_model)
        sub_model = Flatten()(sub_model)

	
        #dynamicLayer = tf.lite.experimental.nn.dynamic_rnn(buildLstmLayer(main_input), time_major=True) 
        #dynamicLayer = tf.lite.experimental.nn.dynamic_rnn(buildLstmLayer(), main_input, dtype="float32") 

        

        # NOTE: I am not passing 'inputs' to the dynamic_rnn(), as i will be passing in model.predict(). IS THIS OK? 
 
        #main_output = Dropout(dropout_rate)(sub_model)
        main_output = sub_model

        return main_output

    if (lstmlast==0):
      first_model = cnn_lstm_cell(main_input1)
      second_model = cnn_lstm_cell(main_input2)
      third_model = cnn_lstm_cell(main_input3)
      model = Concatenate()([first_model, second_model, third_model])  
      #model = Dropout(0.4)(model)
      model = Dense(n_classes)(model)
      #model = BatchNormalization()(model)
      output = Activation('softmax', name="softmax")(model)
    else:
      first_model = cnn_lstm_cell_tflite(main_input1)
      second_model = cnn_lstm_cell_tflite(main_input2)
      third_model = cnn_lstm_cell_tflite(main_input3)
      model = Concatenate()([first_model, second_model, third_model])
      #model = MaxPooling1D(pool_size=pool_size)(model)
      #model = LSTM(128, return_sequences=True)(model)
      #model = LSTM(128)(model)
      #model = LSTM(128, return_sequences=True)(model)
      #model = LSTM(128, return_sequences=True)(model)
      #model = LSTM(128)(model)
      model = Dense(n_classes)(model)
      output = Activation('softmax', name="softmax")(model)
    
    model = Model([main_input1, main_input2, main_input3], output)
    #model = Model(main_input1, output)
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    graph_path = os.path.join(output_path, "merged_model.png")
    plot_model(model, to_file=graph_path, show_shapes=True)  
    metrics = Metrics() 
    history = model.fit([X_trainS1, X_trainS2, X_trainS3], Y_train,
                       batch_size=batch_size,
                        validation_data=([X_valS1, X_valS2, X_valS3], Y_val),
                        epochs=epochs, verbose=1) 

    #model.fit(X_trainS1, Y_train,
    #                    batch_size=batch_size,
    #                    validation_data=(X_valS1,Y_val),
    #                    epochs=epochs, verbose=1) 


    model_path = os.path.join(output_path, "merged_dcl.h5")
    model.save(model_path)

    #print("Max value in weights ")
    #for layer in model.layers:
    #   weights = layer.get_weights() # list of numpy arrays
    #   print(layer.name)
       #print(weights.params[0])
       #w_max = tf.keras.backend.eval(tf.math.reduce_max(weights))
       #print(w_max)
     #  if len(weights)>0:
     #     w_max = tf.keras.backend.eval(tf.math.reduce_max(weights))
     #     #w_max = np.max(weights)
     #     #print(w_max)

    #print("Min value in weights",np.min(model.weights))


    #w_max = tf.keras.backend.eval(tf.math.reduce_max(model.weights))
    #w_min = tf.keras.backend.eval(tf.math.reduce_min(model.weights))

    #print('model weight max',w_max)
    #print('model weight min',w_min)

    print(history.history['val_acc'])
    # summarize history for accuracy
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    # Now we can write out the TensorFlow compatible checkpoint and inference graph which will be used later with the freeze_graph.py script to create the frozen model:
    # make list of output node names
    output_names=[out.op.name for out in model.outputs]
    input_names=[out.op.name for out in model.inputs]

    # make list of output and input node names
    print('Full model input  node is{}'.format(input_names))
    print('Full model output node is{}'.format(output_names))
    print('Total layer count is{} '.format(len(model.layers)))


    print('Generate prediction for sample 100')
    predictions = model.predict([X_trainS1[99:102], X_trainS2[99:102], X_trainS3[99:102]])
    print('predictions shape:', predictions.shape)
    print(*predictions)
    #plt.plot(predictions)
    #plt.show()
    exit()

    print('Generate predictions')
    predictions = model.predict([X_valS1, X_valS2, X_valS3])
    print('predictions shape:', predictions.shape)
    #plt.plot(predictions)
    tf.keras.utils.to_categorical(Y_val, num_classes=6)
    #print(Y_val)
    labels = ['walking', 'walk up', 'walk down', 'sitting', 'standing','laying']

    fig, axs = plt.subplots(2)
    fig.suptitle('Activity predictions (top) vs truth (bottom)')
    lines = axs[0].plot(predictions[200:500,:])
    plt.legend(lines,labels)
    axs[1].plot(Y_val[200: 500 , :])
    
    #lines = plt[0].plot(predictions[0:00,:])
    #plt.legend(lines,labels)
    #plt.title("Salary vs Experience (Testing set)")
    plt.xlabel("Sample number")
    axs[0].set(ylabel='Probability')
    axs[1].set(ylabel='Probability')


    #line1 = plt[1].plot(Y_val[200: 500 , :])
    #plt[1].legend(line1,labels)
    #plt[1].title("Salary vs Experience (Testing set)")
    #plt[1].xlabel("Years of Experience")
    #plt[1].ylabel("Salary")
    plt.show()

        
    #model.summary()



if __name__ == '__main__':
    data = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset")
    output = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset_output")
    create_folder_try(output)
    main(data_path=data, output_path=output)
