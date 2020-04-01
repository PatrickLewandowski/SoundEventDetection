import asyncio
import websockets

import youtube_dl # MUST HAVE youtube-dl cli installed
import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import pandas as pd
from keras.models import load_model
import json

#ML imports
#from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation, Concatenate
from keras.layers import Lambda
#from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Model
import keras.backend as K

#External .py scripts
from lib import mel_features
from lib import vggish_input
from lib import vggish_params
from lib import vggish_postprocess
from lib import vggish_slim
from lib import utilities
from lib import data_generator
from lib.train_functions import evaluateCore, trainCore, average_pooling, max_pooling, attention_pooling, pooling_shape, train, writeToFile
from lib import download_mp3

DIR = os.path.dirname(__file__) + "/"
model = load_model( DIR + "model/"+"200k_DLSA_U"+".h5")
class_labels = pd.read_csv(DIR + "lib/class.csv").drop(columns=['index','mid'])['display_name'].to_numpy()

#Functions
def getLabeling(class_labels,num):
    return class_labels[num]

#Predict on input
def predictOnVggishInput(df, model):
    print('predictOnVggishInput')
    results = []
    attention_results = []
    #Handling edge case of 10 seconds
    length = df.shape[0]
    if(length==10):
        length = 11

    for endNumber in range(10,length):
        startNumber = endNumber - 10
        df_final = [df[startNumber:endNumber]]
        df_final = np.asarray(df_final)
        df_final = (np.float32(df_final) - 128.) / 128.
        output = model.predict(df_final)
        #attention_layer_output=model.get_layer(layer_name).output
        #intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)
        #attention_layer_prediction=intermediate_model.predict(df_final)
        #attention_results.append(attention_layer_prediction)
        results.append(output.astype(np.float32))
    return results

#VGGish transformation
def vggish_fullprocess(wav_file,data,sr,switch):
    print('vggish_fullprocess')
    # wav_file = "Output/dog.wav" #tempSave
    checkpoint = DIR + "model/vggish_model.ckpt"
    pca_params = DIR + "model/vggish_pca_params.npz"

    if switch==1:
        examples_batch = vggish_input.wavfile_to_examples(wav_file)
    else:
        examples_batch = vggish_input.waveform_to_examples(data,sr)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(pca_params)
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],feed_dict={features_tensor: examples_batch})

    postprocessed_batch = pproc.postprocess(embedding_batch)
    return postprocessed_batch

def predictOnAudioFile(wav_file):
    print('predictOnAudioFile') 
    wav_data, sr = sf.read(wav_file, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    if len(samples.shape) > 1:
        data = np.mean(samples, axis=1)
    else:
        data = samples
    data = np.asarray(data)
    result_list = []

    step = int(data.shape[0]/sr)
    #step = 30
    for segment in np.arange(0,int(data.shape[0]/sr),step):
        if((data.shape[0]/sr)-segment<step):
            break

        if segment>0:
            # print(j)
            # await websocket.send(j)
            # shouldSendClasses = F
            segmentStart = segment - 9
        else:
            segmentStart = segment

        data_segment = data[sr*segmentStart:sr*(segment+step)]
        data_embedding = vggish_fullprocess(wav_file,data_segment,sr,2)
        result = predictOnVggishInput(data_embedding, model)
        result_list.append(result)
    return result_list

async def classifier(websocket, path):
    ss = await websocket.recv()
    print(ss)

     fileDestination = DIR + str(ss)
     download_mp3.downloadYoutube(ss,fileDestination + ".f")
     result_list = predictOnAudioFile(fileDestination + ".wav")

     print('#################################')
     print('Detection Finished')

     results = []
     for i in range(0, len(result_list[0])):
         [t]  = result_list[0][i]
         for j in range(0,len(t)):
             if t[j] < 0.25:
                 t[j] = 0.0
             else:
                 t[j] = 1.*t[j]
         t = t.tolist()
         results.append(t)

     results = np.asarray(results)
     results = np.transpose(results)

     print('#################################')
     print('Transformation Finished')


     abc = []
     for i in range(0,len(results)):
         messageMap = {}
         r = results[i].tolist()
         category = getLabeling(class_labels,i)
         messageMap['category'] = category
         messageMap['probabilities'] = r
         abc.append(messageMap)
    
     j = json.dumps(abc)
    

    await websocket.send(j)

start_server = websockets.serve(classifier, '127.0.0.1', 8000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
