import os
import numpy as np
import h5py
import argparse
import time
import logging

import keras
import keras.backend as K
from sklearn import metrics
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda, Activation, Concatenate
#from tensorflow.keras import backend as K

#External .py scripts
from lib import utilities
from lib import data_generator

try:
    import cPickle
except BaseException:
    import _pickle as cPickle

def evaluateCore(args, model, input, target, stats_dir, probs_dir, iteration):
    """Evaluate a model.
    Args:
      model: object
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      stats_dir: str, directory to write out statistics.
      probs_dir: str, directory to write out output (samples_num, classes_num)
      iteration: int
    Returns:
      None
    """

    utilities.create_folder(stats_dir)
    utilities.create_folder(probs_dir)

    # Predict presence probabilittarget
    callback_time = time.time()
    (clips_num, time_steps, freq_bins) = input.shape
    (input, target) = utilities.transform_data(input, target)
    output = model.predict(input)
    output = output.astype(np.float32)  # (clips_num, classes_num)

    # Write out presence probabilities
    #NEW prob_path = os.path.join(probs_dir, "prob_{}_iters.p".format(iteration))
    #NEW cPickle.dump(output, open(prob_path, 'wb'))

    # Calculate statistics
    stats = utilities.calculate_stats(output, target)

    # Write out statistics
    #NEW stat_path = os.path.join(stats_dir, "stat_{}_iters.p".format(iteration))
    #NEW cPickle.dump(stats, open(stat_path, 'wb'))

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    logging.info(
        "mAP: {:.6f}, AUC: {:.6f}, Callback time: {:.3f} s".format(
            mAP, mAUC, time.time() - callback_time))
    
    
    writeToFile(args["filename"], str(mAP),1)
    writeToFile(args["filename"], str(mAUC),1)
    writeToFile(args["filename"], str(time.time() - callback_time),1)
    
    if False:
        logging.info("Saveing prob to {}".format(prob_path))
        logging.info("Saveing stat to {}".format(stat_path))

def trainCore(args):
    """Train a model.
    """

    data_dir = args["data_dir"]
    workspace = args["workspace"]
    mini_data = args["mini_data"]
    balance_type = args["balance_type"]
    learning_rate = args["learning_rate"]
    filename = args["filename"]
    model_type = args["model_type"]
    model = args["model"]
    batch_size = args["batch_size"]

    # Path of hdf5 data
    bal_train_hdf5_path = os.path.join(data_dir, "bal_train.h5")
    unbal_train_hdf5_path = os.path.join(data_dir, "unbal_train.h5")
    test_hdf5_path = os.path.join(data_dir, "eval.h5")

    # Load data
    load_time = time.time()

    if mini_data:
        # Only load balanced data
        (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
            bal_train_hdf5_path)

        train_x = bal_train_x
        train_y = bal_train_y
        train_id_list = bal_train_id_list

    else:
        # Load both balanced and unbalanced data
        (bal_train_x, bal_train_y, bal_train_id_list) = utilities.load_data(
            bal_train_hdf5_path)

        (unbal_train_x, unbal_train_y, unbal_train_id_list) = utilities.load_data(
            unbal_train_hdf5_path)

        train_x = np.concatenate((bal_train_x, unbal_train_x))
        train_y = np.concatenate((bal_train_y, unbal_train_y))
        train_id_list = bal_train_id_list + unbal_train_id_list

    # Test data
    (test_x, test_y, test_id_list) = utilities.load_data(test_hdf5_path)

    logging.info("Loading data time: {:.3f} s".format(time.time() - load_time))
    logging.info("Training data shape: {}".format(train_x.shape))

    # Optimization method
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Output directories
    sub_dir = os.path.join(filename,
                           'balance_type={}'.format(balance_type),
                           'model_type={}'.format(model_type))
    
    models_dir = os.path.join(workspace, "models", sub_dir)
    utilities.create_folder(models_dir)

    stats_dir = os.path.join(workspace, "stats", sub_dir)
    utilities.create_folder(stats_dir)

    probs_dir = os.path.join(workspace, "probs", sub_dir)
    utilities.create_folder(probs_dir)

    # Data generator
    if balance_type == 'no_balance':
        DataGenerator = data_generator.VanillaDataGenerator

    elif balance_type == 'balance_in_batch':
        DataGenerator = data_generator.BalancedDataGenerator

    else:
        raise Exception("Incorrect balance_type!")

    train_gen = DataGenerator(
        x=train_x,
        y=train_y,
        batch_size=batch_size,
        shuffle=True,
        seed=1234)

    iteration = 0
    call_freq = 1000
    train_time = time.time()
    
    writeToFile(args["filename"], "iteration,traintTime,trainmAP,trainAUC,trainCallbackTime,testmAP,testAUC,testCallbackTime,None\n",0)
    
    for (batch_x, batch_y) in train_gen.generate():

        # Compute stats every several interations
        if iteration % call_freq == 0:
            logging.info("------------------")

            logging.info(
                "Iteration: {}, train time: {:.3f} s".format(
                    iteration, time.time() - train_time))
            
            writeToFile(args["filename"], str(iteration),1)
            writeToFile(args["filename"], str(time.time() - train_time),1)
            
            logging.info("Balance train statistics:")
            evaluateCore(
		args=args,
                model=model,
                input=bal_train_x,
                target=bal_train_y,
                stats_dir=os.path.join(stats_dir, 'bal_train'),
                probs_dir=os.path.join(probs_dir, 'bal_train'),
                iteration=iteration
            )

            logging.info("Test statistics:")
            evaluateCore(
		args=args,
                model=model,
                input=test_x,
                target=test_y,
                stats_dir=os.path.join(stats_dir, "test"),
                probs_dir=os.path.join(probs_dir, "test"),
                iteration=iteration
            )
            
            writeToFile(args["filename"], '\n',0)
            train_time = time.time()

        # Update params
        (batch_x, batch_y) = utilities.transform_data(batch_x, batch_y)
        model.train_on_batch(x=batch_x, y=batch_y)

        iteration += 1
        
        # Save model
        save_out_path = os.path.join(
            models_dir, "md_{}_iters.h5".format(iteration))
            
        #NEW
        if (iteration%2000 == 0):
            model.save(save_out_path)
        
        # Stop training when maximum iteration achieves
        if iteration == 200001:
            break

def average_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.mean(input, axis=1)


def max_pooling(inputs, **kwargs):
    input = inputs[0]   # (batch_size, time_steps, freq_bins)
    return K.max(input, axis=1)


def attention_pooling(inputs, **kwargs):
    [out, att] = inputs

    epsilon = 1e-7
    att = K.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)
  
def pooling_shape(input_shape):
    if isinstance(input_shape, list):
        (sample_num, time_steps, freq_bins) = input_shape[0]

    else:
        (sample_num, time_steps, freq_bins) = input_shape

    return (sample_num, freq_bins)

def train(args, option=1):
    
    model_type = args["model_type"]

    time_steps = 10
    freq_bins = 128
    classes_num = 527

    # Hyper parameters
    hidden_units = 1024
    drop_rate = 0.5
    batch_size = 500
    
    # Embedded layers
    input_layer = Input(shape=(time_steps, freq_bins))
    
    a1 = Dense(hidden_units)(input_layer)
    a1 = BatchNormalization()(a1)
    a1 = Activation('relu')(a1)
    a1 = Dropout(drop_rate)(a1)
    
    a2 = Dense(hidden_units)(a1)
    a2 = BatchNormalization()(a2)
    a2 = Activation('relu')(a2)
    a2 = Dropout(drop_rate)(a2)
    
    a3 = Dense(hidden_units)(a2)
    a3 = BatchNormalization()(a3)
    a3 = Activation('relu')(a3)
    a3 = Dropout(drop_rate)(a3)
    
    # Pooling layers
    if model_type == 'decision_level_max_pooling':
        '''Global max pooling.
        
        [1] Choi, Keunwoo, et al. "Automatic tagging using deep convolutional 
        neural networks." arXiv preprint arXiv:1606.00298 (2016).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        output_layer = Lambda(max_pooling, output_shape=pooling_shape)([cla])

    elif model_type == 'decision_level_average_pooling':
        '''Global average pooling.
        
        [2] Lin, Min, et al. Qiang Chen, and Shuicheng Yan. "Network in 
        network." arXiv preprint arXiv:1312.4400 (2013).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        output_layer = Lambda(
            average_pooling,
            output_shape=pooling_shape)(
            [cla])

    elif model_type == 'custom':
        '''Our model.
        
        [2] Lin, Min, et al. Qiang Chen, and Shuicheng Yan. "Network in 
        network." arXiv preprint arXiv:1312.4400 (2013).
        '''
        cla = Dense(classes_num, activation='softmax')(a3)
        output_layer = Lambda(
            gaussian_normalization,
            output_shape=pooling_shape)(
            [cla])
        
        
        
    elif model_type == 'decision_level_single_attention':
        '''Decision level single attention pooling.
        [3] Kong, Qiuqiang, et al. "Audio Set classification with attention
        model: A probabilistic perspective." arXiv preprint arXiv:1711.00927
        (2017).
        '''
        cla = Dense(classes_num, activation='sigmoid')(a3)
        att = Dense(classes_num, activation='softmax')(a3)
        output_layer = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla, att])

    elif model_type == 'decision_level_multi_attention':
        '''Decision level multi attention pooling.
        [4] Yu, Changsong, et al. "Multi-level Attention Model for Weakly
        Supervised Audio Classification." arXiv preprint arXiv:1803.02353
        (2018).
        '''
        cla1 = Dense(classes_num, activation='sigmoid')(a2)
        att1 = Dense(classes_num, activation='softmax')(a2)
        out1 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla1, att1])

        cla2 = Dense(classes_num, activation='sigmoid')(a3)
        att2 = Dense(classes_num, activation='softmax')(a3)
        out2 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla2, att2])

        b1 = Concatenate(axis=-1)([out1, out2])
        b1 = Dense(classes_num)(b1)
        output_layer = Activation('sigmoid')(b1)

    elif model_type == 'feature_level_attention':
        '''Feature level attention.
        [1] To be appear.
        '''
        cla = Dense(hidden_units, activation='linear')(a3)
        att = Dense(hidden_units, activation='sigmoid')(a3)
        b1 = Lambda(
            attention_pooling, output_shape=pooling_shape)([cla, att])

        b1 = BatchNormalization()(b1)
        b1 = Activation(activation='relu')(b1)
        b1 = Dropout(drop_rate)(b1)

        output_layer = Dense(classes_num, activation='sigmoid')(b1)

    else:
        raise Exception("Incorrect model_type!")
    
    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    args["model"] = model
    args["batch_size"] = batch_size

    if(option==0):
      # Train
      trainCore(args)
    elif(option==1):
      # Load
      return model

def writeToFile(fileName,string, mode):
    with open(fileName+".csv", "a") as f:
        if(mode == 0):
            f.write("".join(string))
        elif(mode == 1):
            f.write("".join(string+","))


