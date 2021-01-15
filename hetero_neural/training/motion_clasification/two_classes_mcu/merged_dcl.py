# -- coding: utf-8 --

import os
#os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

import sys
import tensorflow as tf
#import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Dense, Activation, Concatenate, Flatten
from tensorflow.keras.utils import to_categorical,plot_model
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


lstmlast = 1

current_activity = 5
train_data = 3
val_data = 3


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

    print(Y_train[0])
    #Y_train[Y_train > 1] = 2

    Y_train[Y_train == current_activity] = 7

    Y_train[Y_train < 7] = 1

    Y_train[Y_train == 7] = 2

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

    #Y_val[Y_val > 1] = 2


    Y_val[Y_val == current_activity] = 7

    Y_val[Y_val < 7] = 1

    Y_val[Y_val == 7] = 2


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

    if(train_data == 1):
       X_train_data =  X_trainS1
    elif(train_data == 2):
       X_train_data =  X_trainS2
    else:
       X_train_data =  X_trainS3

    if(val_data == 1):
       X_val_data =  X_valS1
    elif(val_data == 2):
       X_val_data =  X_valS2
    else:
       X_val_data =  X_valS3

    #X_train_data = X_train_data.as_matrix()
    scaler = MinMaxScaler()
    #X_train_data = scaler.fit_transform(X_train_data)


    X_train_data = scaler.fit_transform(X_train_data.reshape(-1, X_train_data.shape[-1])).reshape(X_train_data.shape)

    #print(X_train_data.shape)

    X_train_data_q = (X_train_data * 255).astype(int)
 


    X_train_data = X_train_data_q/255

    #X_train_data_q = X_train_data_q.transpose([1, 2, 0])

    print(X_train_data_q.shape)

    #select sample 100 row, all the 128 column and a particular depth value 

    savetxt('./X_train_data_q_x.csv', X_train_data_q[100,:,0], newline=',',delimiter=',',fmt='%d')

    #savetxt('./X_train_data_q_x.csv', X_train_data_q[:,0,0], delimiter=',',fmt='%d')

    savetxt('./X_train_data_q_y.csv', X_train_data_q[100,:,1], newline=',',delimiter=',',fmt='%d')

    savetxt('./X_train_data_q_z.csv', X_train_data_q[100,:,2], newline=',',delimiter=',',fmt='%d')

    X_val_data = scaler.fit_transform(X_val_data.reshape(-1, X_val_data.shape[-1])).reshape(X_val_data.shape)

    #print(X_val_data.shape)

    X_val_data_q = (X_val_data * 255).astype(int)

    X_val_data = X_val_data_q/255

    #print(X_train_data[0])
    print(X_train_data_q[0])

    #print(Y_train[100])
    print(Y_train[0])
    #exit()

    epochs = 100
    batch_size = 256
    kernel_size = 3
    pool_size = 2
    dropout_rate = 0.15
    n_classes = 2

    f_act = 'relu'


    main_input1 = Input(shape=(128, 3), name='main_input1')
    #main_input2 = Input(shape=(128, 3), name='main_input2')
    #main_input3 = Input(shape=(128, 3), name='main_input3')

    def cnn_lstm_cell(main_input):
        sub_model = Conv1D(64, kernel_size, input_shape=(128, 3), activation=f_act, padding='same')(main_input)
        sub_model = BatchNormalization()(sub_model)
        sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        #sub_model = Dropout(dropout_rate)(sub_model)
        sub_model = Conv1D(16, kernel_size, activation=f_act, padding='same')(sub_model)
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
        sub_model = Conv1D(8, kernel_size, input_shape=(128, 3), activation='relu', padding='same')(main_input)
        #sub_model = Conv1D(8, kernel_size, input_shape=(128, 3), activation=f_act, padding='same')(main_input)
        ##sub_model = BatchNormalization()(sub_model)
        #sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        ##sub_model = Dropout(dropout_rate)(sub_model)
        #sub_model = Conv1D(4, kernel_size, activation=f_act, padding='same')(sub_model)
        ##sub_model = BatchNormalization()(sub_model)
        #sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
        ##sub_model = Dropout(dropout_rate)(sub_model)
        #sub_model = Conv1D(32, kernel_size, activation=f_act, padding='same')(sub_model)
        ##sub_model = BatchNormalization()(sub_model)
        #sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
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
      #second_model = cnn_lstm_cell_tflite(main_input2)
      #third_model = cnn_lstm_cell_tflite(main_input3)
      #model = Concatenate()([first_model, second_model, third_model])
      #model = MaxPooling1D(pool_size=pool_size)(model)
      #model = LSTM(128, return_sequences=True)(model)
      #model = LSTM(128)(model)
      #model = LSTM(128, return_sequences=True)(model)
      #model = LSTM(128, return_sequences=True)(model)
      #model = LSTM(128)(model)
      #model = Dense(10)(first_model)
      model = Dense(2)(first_model)
      output = Activation('softmax', name="softmax")(model)
    
    #model = Model([main_input1, main_input2, main_input3], output)
    model = Model(main_input1, output)
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    graph_path = os.path.join(output_path, "merged_model.png")
    plot_model(model, to_file=graph_path, show_shapes=True)  
    metrics = Metrics() 
    #history = model.fit([X_trainS1, X_trainS2, X_trainS3], Y_train,
    #                   batch_size=batch_size,
    #                    validation_data=([X_valS1, X_valS2, X_valS3], Y_val),
    #                    epochs=epochs, verbose=1) 

    #Y_train[Y_train > 1] = 2
    #Y_val[Y_val > 1] = 2

    #print(Y_train)

    #history = model.fit(X_trainS1, Y_train,
    #                    batch_size=batch_size,
    #                    validation_data=(X_valS1,Y_val),
    #                    epochs=epochs, verbose=1) 

    #history = model.fit(X_trainS3, Y_train,
    #                    batch_size=batch_size,
    #                    validation_data=(X_valS3,Y_val),
    #                    epochs=epochs, verbose=1)

    #history = model.fit(X_train_data, Y_train,
    #                    batch_size=batch_size,
    #                    validation_data=(X_val_data,Y_val),
    #                    epochs=epochs, verbose=1)

    history = model.fit(X_train_data, Y_train,
                        batch_size=batch_size,
                        validation_data=(X_val_data,Y_val),
                        epochs=epochs, verbose=1)


    model_path = os.path.join(output_path, "merged_dcl.h5")
    model.save(model_path)

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

    print('Generate predictions for 300 samples')
    predictions = model.predict(X_val_data)
    print(predictions[0])
    print(predictions[100])

    #exit()
    print('predictions shape:', predictions.shape)
    tf.keras.utils.to_categorical(Y_val, num_classes=2)
    labels = ['current', 'other']

    fig, axs = plt.subplots(2)
    fig.suptitle('Activity predictions (top) vs truth (bottom)')
    lines = axs[0].plot(predictions[0:300,:])
    plt.legend(lines,labels)
    axs[1].plot(Y_val[0: 300 , :])
    
    plt.xlabel("Sample number")
    axs[0].set(ylabel='Probability')
    axs[1].set(ylabel='Probability')

    plt.show()




if __name__ == '__main__':

    data = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset")
    output = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset_output")
    create_folder_try(output)
    main(data_path=data, output_path=output)
