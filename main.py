import tensorflow as tf
from preprocess import *
from tensorflow.keras.models import load_model


def inference(filename, type):
    data = np.load(filename)
    
    print(type)
    
    if type=='CWT-TF':
        dataset = loads_cwt(data)
        wavelet = "mexh"  # mexh, morl, gaus8, gaus4
        scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)
        x1, x2 = worker_cwt(dataset, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate)
        x1_test = np.reshape(x1, (len(x1), 100, 100, 1))
        model=load_model('wavelet.h5')
        y = model.predict(x1_test)
        y = y.round()
        
    elif type=='DWT-TF':
        x_train = worker_dwt(data)
        x_train_arr = np.array(x_train)
        x1_test = tf.stack(x_train_arr)
        model=load_model('wavelet_dwt.h5')
        y = model.predict(x1_test)
        y = y.round()
        
    elif type=='SWT-TF':
        x_train = worker_swt(data)
        x_train_arr = np.array(x_train)
        x_train_arr = np.reshape(x_train_arr, (x_train_arr.shape[0], 180, 2))
        x1_test = tf.stack(x_train_arr)
        model=load_model('wavelet_swt (1).h5')
        y = model.predict(x1_test)
        y = y.round()
    
    elif type=='CWT-TFLITE':
        dataset = loads_cwt(data)
        wavelet = "mexh"  # mexh, morl, gaus8, gaus4
        scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)
        x1, x2 = worker_cwt(dataset, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate)
        x1_test = np.reshape(x1, (len(x1), 100, 100, 1))
        
        interpreter = tf.lite.Interpreter(model_path="wavelet_cwt.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        output = []
        for i in x1_test:
            input_data = np.float32(i)
            input_data = np.reshape(input_data, (1, 100, 100, 1))
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            output.append(output_data)
        
        y = np.array(output).reshape(-1, 5).round()


    elif type=='DWT-TFLITE':
        x_train = worker_dwt(data)
        x_train_arr = np.array(x_train)
        x1_test = tf.stack(x_train_arr)
        
        interpreter = tf.lite.Interpreter(model_path="wavelet_dwt.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        output = []
        for i in x1_test:
            input_data = np.float32(i)
            input_data = np.reshape(input_data, (1, 32, 1))
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            output.append(output_data)

        y = np.array(output).reshape(-1, 5).round()
        
    else:
        x_train = worker_swt(data)
        x_train_arr = np.array(x_train)
        x_train_arr = np.reshape(x_train_arr, (x_train_arr.shape[0], 180, 2))
        x1_test = tf.stack(x_train_arr)
        
        interpreter = tf.lite.Interpreter(model_path="wavelet_swt (1).tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        output = []
        for i in x1_test:
            input_data = np.float32(i)
            input_data = np.reshape(input_data, (1, 180, 2))
            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output.append(output_data)
        
        y = np.array(output).reshape(-1, 5).round()

    
    y_label = np.argmax(y, axis=1)

    # map to class
    N, S, V, F, Q = 0, 0, 0, 0, 0
    for i in y_label:
        if i == 0:
            N += 1
        elif i == 1:
            S += 1
        elif i == 2:
            V += 1
        elif i == 3:
            F += 1
        elif i == 4:
            Q += 1

    # print result 
    print("N: ", N)
    print("S: ", S)
    print("V: ", V)
    print("F: ", F)
    print("Q: ", Q)
    
    return N, S, V, F, Q