
from typing import Dict, OrderedDict
from attr import set_run_validators
from matplotlib.collections import Collection
import nest_asyncio
from sklearn.metrics import log_loss
nest_asyncio.apply()
import collections
from csv import writer
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds
import pandas as pd
import gc

import random
NUM_CLIENTS = 48
NUM_CLIENTS_PER_SERVER = 12
NUM_EPOCHS_EMNIST = 10
NUM_EPOCHS_fashion = 10
NUM_EPOCHS_svhn = 10
NUM_EPOCHS_CIFAR10 = 10
BATCH_SIZE = 100
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS =100
VARIANCE = 0.2
randomFEDAVG = False
MATCH = True
def main():
    
    #number of splits to divide datasets into clients
    split = 600

    np.random.seed(0)
    
    svhn_train=tfds.load("svhn_cropped", split= 'train')
    svhn_test=tfds.load("svhn_cropped", split= 'test')
    
    svhn_train_x=[]
    svhn_train_y=[]
    #take an image in the dataset along with the label and put them in x y 
    for element in svhn_train.take(-1):
        svhn_train_x.append(tfds.as_numpy(element['image']).astype(np.float32))
        svhn_train_y.append(tfds.as_numpy(element['label']).astype(np.int32))

    svhn_total_image_count = len(svhn_train_x)
    #determine number of images in a split
    svhn_image_per_set = int(np.floor(svhn_total_image_count/split))

    #Dict to hold dataset eventually
    svhn_client_train_dataset = collections.OrderedDict()

    #Assign images to cliens, using variance for different client dataset sizes
    end =-1

    for i in range(1, split+1):
        #set client name
        client_name = "client_" + str(i)
        #begin dataset after the end of previous dataset
        start = end + 1
        #determine size of dataset +/- variance
        end = end + svhn_image_per_set+round(svhn_image_per_set*(2*(np.random.rand()-0.5)*VARIANCE)) 
        #if, through variance, index would be out of bounds, set the end to maximum index
        if(end>len(svhn_train)-1): end = len(svhn_train_x) - 1
        print(f"Adding data from {start} to {end} for client : {client_name}")
        #create a dict line for the client dataset
        data = collections.OrderedDict((('label', svhn_train_y[start:end]), ('pixels', svhn_train_x[start:end])))
        svhn_client_train_dataset[client_name] = data


    


    cifar10_train = tfds.load("cifar10", split = 'train')
    cifar10_test = tfds.load("cifar10", split = 'test')
    
   

    cifar10_client_train_dataset = collections.OrderedDict()
    cifar10_train_x=[]
    cifar10_train_y=[]
    for element in cifar10_train.take(-1):
        cifar10_train_x.append(tfds.as_numpy(element['image']).astype(np.float32))
        cifar10_train_y.append(tfds.as_numpy(element['label']).astype(np.int32))

    cifar10_total_image_count = len(cifar10_train_x)
    cifar10_image_per_set = int(np.floor(cifar10_total_image_count/split))


    end =-1
    for i in range(1, split+1): 
        client_name = "client_" + str(i)
        start = end + 1
        end = end + cifar10_image_per_set+round(cifar10_image_per_set*(2*(np.random.rand()-0.5)*VARIANCE))
        if(end>len(cifar10_train)-1): end = len(cifar10_train_x) - 1
        print(f"Adding data from {start} to {end} for client : {client_name}")
        data = collections.OrderedDict((('label', cifar10_train_y[start:end]), ('pixels', cifar10_train_x[start:end])))
        cifar10_client_train_dataset[client_name] = data


    
   
    (fashion_train_x, fashion_train_y), (x_test, y_test)  = tf.keras.datasets.fashion_mnist.load_data()

    fashion_train_x = fashion_train_x.astype(np.float32)
    fashion_train_y = fashion_train_y.astype(np.int32)
    x_test = x_test.astype(np.float32).reshape(10000, 28, 28, 1)
    y_test = y_test.astype(np.int32).reshape(10000, 1)

    fashion_total_image_count = len(fashion_train_x)
    fashion_image_per_set = int(np.floor(fashion_total_image_count/split))

    cifar100_train, cifar100_test = tff.simulation.datasets.cifar100.load_data()

    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    emnist_client_train_dataset = collections.OrderedDict()
    i = 1
    for dataset in emnist_train.datasets(limit_count = NUM_CLIENTS):
        client_name = "client_" + str(i)
        print(f"Adding data for client : {client_name}")
        ys = []
        xs = []
        for element in dataset.as_numpy_iterator():
            ys.append(element['label'])
            xs.append(element['pixels'])
        emnist_client_train_dataset[client_name] = collections.OrderedDict((('label', ys), ('pixels', xs)))
        i =i+1


    fashion_client_train_dataset = collections.OrderedDict()
    end =-1
    for i in range(1, split+1):
        client_name = "client_" + str(i)
        start = end + 1
        end = end + fashion_image_per_set+round(fashion_image_per_set*(2*(np.random.rand()-0.5)*VARIANCE))
        if(end>len(fashion_train_x)-1): end = len(fashion_train_x) - 1
        print(f"Adding data from {start} to {end} for client : {client_name}")
        data = collections.OrderedDict((('label', fashion_train_y[start:end]), ('pixels', fashion_train_x[start:end])))
        fashion_client_train_dataset[client_name] = data

    

    #create federated datasets from client datasets
    cifar10_data = tff.simulation.datasets.TestClientData(cifar10_client_train_dataset)                 
    fashion_data = tff.simulation.datasets.TestClientData(fashion_client_train_dataset)
    svhn_data = tff.simulation.datasets.TestClientData(svhn_client_train_dataset)
    emnist_data = tff.simulation.datasets.TestClientData(emnist_client_train_dataset)
    #Additional shaping for models
    cifar10_dataset = cifar10_data.create_tf_dataset_for_client(cifar10_data.client_ids[0])
    svhn_dataset = svhn_data.create_tf_dataset_for_client(svhn_data.client_ids[0])
    fashion_dataset = fashion_data.create_tf_dataset_for_client(fashion_data.client_ids[0])
    emnist_dataset = emnist_data.create_tf_dataset_for_client(emnist_data.client_ids[0])
  
    global preprocessed_cifar10_dataset 
    global preprocessed_emnist_dataset 
    global preprocessed_fashion_dataset 
    global preprocessed_svhn_dataset

    preprocessed_cifar10_dataset = preprocesscifar10(cifar10_dataset)
    preprocessed_emnist_dataset = preprocessemnist(emnist_dataset)
    preprocessed_fashion_dataset = preprocessfashion(fashion_dataset)
    preprocessed_svhn_dataset = preprocesssvhn(svhn_dataset)

  


    cifar10_iterative_process = tff.learning.build_federated_averaging_process(
    modelcifar10_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1))

    fashion_iterative_process = tff.learning.build_federated_averaging_process(
    modelfashion_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1))


    emnist_iterative_process = tff.learning.build_federated_averaging_process(
    modelemnist_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1))

    svhn_iterative_process = tff.learning.build_federated_averaging_process(
    modelsvhn_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1))

    
    
    
    cifar10state = cifar10_iterative_process.initialize()
    fashionstate = fashion_iterative_process.initialize()
    emniststate  = emnist_iterative_process.initialize()
    svhnstate = svhn_iterative_process.initialize()
    
    cifar10_evaluation = tff.learning.build_federated_evaluation(modelcifar10_fn)
    fashion_evaluation = tff.learning.build_federated_evaluation(modelfashion_fn)
    emnist_evaluation = tff.learning.build_federated_evaluation(modelemnist_fn)
    svhn_evaluation = tff.learning.build_federated_evaluation(modelsvhn_fn)


    cifar10acc = []
    fashionacc = []
    emnistacc = []
    svhnacc = []

    clientlist = []
    for i in range(1, NUM_CLIENTS+1):
        clientlist.append("client_" + str(i))

    for round_num in range(1, NUM_ROUNDS):
        "for each server: poll each client to find the loss, rank, asssign random incentive, execute matching"
        round_client_list_cifar10 = []
        round_client_list_emnist = []
        round_client_list_fashion = []
        round_client_list_svhn = []

        if(randomFEDAVG ==True):
            roundclientlist = clientlist.copy()
            random.shuffle(roundclientlist)
            while(len(roundclientlist)>0):
                
                random.shuffle(roundclientlist)
                round_client_list_cifar10.append(roundclientlist.pop())
                round_client_list_emnist.append(roundclientlist.pop())
                round_client_list_fashion.append(roundclientlist.pop())
                round_client_list_svhn.append(roundclientlist.pop())
            federated_train_data_svhn = make_federated_data_svhn(svhn_data, round_client_list_svhn)
            federated_train_data_emnist = make_federated_data_emnist(emnist_data, round_client_list_emnist)
            federated_train_data_fashion = make_federated_data_fashion(fashion_data, round_client_list_fashion)
            federated_train_data_cifar10 = make_federated_data_cifar10(cifar10_data, round_client_list_cifar10)
        
        
        if(randomFEDAVG==False and MATCH == True):
            
            
            federated_train_data_svhn = make_federated_data_svhn(svhn_data, clientlist)
            overallLosssvhn = svhn_evaluation(svhnstate.model, federated_train_data_svhn)

            federated_train_data_emnist = make_federated_data_emnist(emnist_data, clientlist)
            overallLossemnist = emnist_evaluation(emniststate.model, federated_train_data_emnist)

            federated_train_data_fashion = make_federated_data_fashion(fashion_data, clientlist)
            overallLossfashion = fashion_evaluation(fashionstate.model, federated_train_data_fashion)

            federated_train_data_cifar10 = make_federated_data_cifar10(cifar10_data, clientlist)
            overallLosscifar10 = cifar10_evaluation(cifar10state.model, federated_train_data_cifar10)

            clientLosssvhn = []
            clientLossemnist= []
            clientLossfashion = []
            clientLosscifar10 = []
            for  client in clientlist:
                federated_train_data_svhn = make_federated_data_svhn(svhn_data, [client])
                eval = svhn_evaluation(svhnstate.model, federated_train_data_svhn)
                clientLosssvhn.append([client, eval])
            for  client in clientlist:
                federated_train_data_emnist = make_federated_data_emnist(emnist_data, [client])
                eval = emnist_evaluation(emniststate.model, federated_train_data_emnist)
                clientLossemnist.append([client, eval])
            for  client in clientlist:
                federated_train_data_fashion = make_federated_data_fashion(fashion_data, [client])
                eval = fashion_evaluation(fashionstate.model, federated_train_data_fashion)
                clientLossfashion.append([client, eval])
            for  client in clientlist:
                federated_train_data_cifar10 = make_federated_data_cifar10(cifar10_data, [client])
                eval = cifar10_evaluation(cifar10state.model, federated_train_data_cifar10)
                clientLosscifar10.append([client, eval])

            svhnLossList=[]
            emnistLossList=[]
            fashionLossList=[]
            cifar10LossList=[]
            for client in clientLosssvhn:
                svhnLossList.append([client[0],[client[1]['eval']['loss']-overallLosssvhn['eval']['loss']],[client[1]['eval']['num_examples']]])
            for client in clientLossemnist:
                emnistLossList.append([client[0],[client[1]['eval']['loss']-overallLossemnist['eval']['loss']],[client[1]['eval']['num_examples']]])
            for client in clientLossfashion:
                fashionLossList.append([client[0],[client[1]['eval']['loss']-overallLossfashion['eval']['loss']],[client[1]['eval']['num_examples']]])
            for client in clientLosscifar10:
                cifar10LossList.append([client[0],[client[1]['eval']['loss']-overallLosscifar10['eval']['loss']],[client[1]['eval']['num_examples']]])


            svhnSortedList = sorted(svhnLossList, key = lambda line: line[1],reverse = True)
            emnistSortedList = sorted(emnistLossList, key = lambda line: line[1],reverse = True)
            fashionSortedList = sorted(fashionLossList, key = lambda line: line[1],reverse = True)
            cifar10SortedList = sorted(cifar10LossList, key = lambda line: line[1],reverse = True)

            clientPref = []
            for i in range(0, len(svhnLossList)):
                svhn = ['svhn',svhnLossList[i][2]]
                emnist = ['emnist',emnistLossList[i][2]]
                fashion = ['fashion', fashionLossList[i][2]]
                cifar10 = ['cifar10',cifar10LossList[i][2]]
                serverList = []
                serverList.append(svhn)
                serverList.append(emnist)
                serverList.append(fashion)
                serverList.append(cifar10)
                serverList.sort(key = lambda line:line[1])
                clientPref.append([svhnLossList[i][0],serverList])

                



            round_client_list_cifar10 = []
            round_client_list_emnist = []
            round_client_list_fashion = []
            round_client_list_svhn = []
            matchedsvhn, matchedcifar10, matchfashion, matchedemnist = match(clientPref, svhnSortedList,cifar10SortedList,fashionSortedList,emnistSortedList)




            federated_train_data_svhn = make_federated_data_svhn(svhn_data, matchedsvhn)
            federated_train_data_emnist = make_federated_data_emnist(emnist_data, matchedemnist)
            federated_train_data_fashion = make_federated_data_fashion(fashion_data, matchfashion)
            federated_train_data_cifar10 = make_federated_data_cifar10(cifar10_data, matchedcifar10)
      
        """
        cifar10preflist = np.array(cifar10SortedList).T[0].tolist()
        emnistpreflist = np.array(emnistSortedList).T[0].tolist()
        fashionpreflist = np.array(fashionSortedList).T[0].tolist()
        svhnpreflist = np.array(svhnSortedList).T[0].tolist()
        """
        """
        while len(svhnpreflist)>0:
            clientToBeAdded = (cifar10preflist.pop(0))
            round_client_list_cifar10.append(clientToBeAdded)
            emnistpreflist.remove(clientToBeAdded)
            fashionpreflist.remove(clientToBeAdded)
            svhnpreflist.remove(clientToBeAdded)

            clientToBeAdded = (emnistpreflist.pop(0))
            round_client_list_emnist.append(clientToBeAdded)
            svhnpreflist.remove(clientToBeAdded)
            fashionpreflist.remove(clientToBeAdded)
            cifar10preflist.remove(clientToBeAdded)

            clientToBeAdded = (fashionpreflist.pop(0))
            round_client_list_fashion.append(clientToBeAdded)
            emnistpreflist.remove(clientToBeAdded)
            svhnpreflist.remove(clientToBeAdded)
            cifar10preflist.remove(clientToBeAdded)

            clientToBeAdded = (svhnpreflist.pop(0))
            round_client_list_svhn.append(clientToBeAdded)
            emnistpreflist.remove(clientToBeAdded)
            fashionpreflist.remove(clientToBeAdded)
            cifar10preflist.remove(clientToBeAdded)
   
       """

 
        cifar10state, cifar10metrics = cifar10_iterative_process.next(cifar10state, federated_train_data_cifar10)
        emniststate, emnistmetrics = emnist_iterative_process.next(emniststate, federated_train_data_emnist)
        fashionstate, fashionmetrics = fashion_iterative_process.next(fashionstate, federated_train_data_fashion)
        svhnstate, svhnmetrics = svhn_iterative_process.next(svhnstate, federated_train_data_svhn)

        print('round {:2d}, emnistmetrics={}'.format(round_num, emnistmetrics))
        print('round {:2d}, fashionmetrics={}'.format(round_num, fashionmetrics))
        print('round {:2d}, svhnmetrics={}'.format(round_num, svhnmetrics))
        print('round {:2d}, cifar10={}'.format(round_num, cifar10metrics))

        fashionacc  = ([float(fashionmetrics['train']['sparse_categorical_accuracy'])])
        emnistacc = ([float(emnistmetrics['train']['sparse_categorical_accuracy'])])
        svhnacc = ([float(svhnmetrics['train']['sparse_categorical_accuracy'])])
        cifar10acc = ([float(cifar10metrics['train']['sparse_categorical_accuracy'])])
        
        with open('fashion.csv','a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(fashionacc)
            f_object.close()
        with open('emnist.csv','a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(emnistacc)
            f_object.close()
        with open('svhn.csv','a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(svhnacc)
            f_object.close()
        with open('cifar10.csv','a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(cifar10acc)
            f_object.close()

def convert(l):
    it = iter(l)
    ret_dct = dict(zip(it, it, it))
    return ret_dct
def create_keras_emnist_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28,28,1)),
      
      tf.keras.layers.Conv2D(32, 3),
      tf.keras.layers.BatchNormalization(),
      
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Conv2D(64, 3),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Conv2D(128, 3),
      
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(256, 3),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128),
      
      tf.keras.layers.Dense(10,kernel_regularizer='l1'),
      tf.keras.layers.Softmax(),
  ])

def create_keras_fashion_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28,28,1)),
      
      tf.keras.layers.Conv2D(32, 3),
      tf.keras.layers.BatchNormalization(),
      
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Conv2D(64, 3),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Conv2D(128, 3),
      
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(256, 3),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128),
      
      tf.keras.layers.Dense(10,kernel_regularizer='l1'),
      tf.keras.layers.Softmax(),
  ])
  
def create_keras_svhn_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(32,32,3)),
      
      tf.keras.layers.Conv2D(32, 3),
      tf.keras.layers.BatchNormalization(),
      
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Conv2D(64, 3),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Conv2D(128, 3),
      
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(256, 3),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128),
      
      tf.keras.layers.Dense(10,kernel_regularizer='l1'),
      tf.keras.layers.Softmax(),
  ])

 
def  create_keras_cifar10_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(32,32,3)),
      
      tf.keras.layers.Conv2D(32, 3),
      tf.keras.layers.BatchNormalization(),
      
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Conv2D(64, 3),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Conv2D(128, 3),
      
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(256, 3),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128),
      
      tf.keras.layers.Dense(10,kernel_regularizer='l1'),
      tf.keras.layers.Softmax(),
  ])
def modelemnist_fn():
# We _must_ create a new model here, and _not_ capture it from an external
# scope. TFF will call this within different graph contexts.
    
    keras_model = create_keras_emnist_model()
    return tff.learning.from_keras_model(
    keras_model,
    input_spec=preprocessed_emnist_dataset.element_spec,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

def modelfashion_fn():
# We _must_ create a new model here, and _not_ capture it from an external
# scope. TFF will call this within different graph contexts.
    
    keras_model = create_keras_fashion_model()
    return tff.learning.from_keras_model(
    keras_model,
    input_spec=preprocessed_fashion_dataset.element_spec,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

def modelsvhn_fn():
# We _must_ create a new model here, and _not_ capture it from an external
# scope. TFF will call this within different graph contexts.
    
    keras_model = create_keras_svhn_model()
    return tff.learning.from_keras_model(
    keras_model,
    input_spec=preprocessed_svhn_dataset.element_spec,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])    


def modelcifar10_fn():
# We _must_ create a new model here, and _not_ capture it from an external
# scope. TFF will call this within different graph contexts.
    
    keras_model = create_keras_cifar10_model()
    return tff.learning.from_keras_model(
    keras_model,
    input_spec=preprocessed_cifar10_dataset.element_spec,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])    

def preprocesscifar10(dataset):

  def cifar10_batch_format_fn(element):
    
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 32, 32,3]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS_CIFAR10).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(cifar10_batch_format_fn).prefetch(PREFETCH_BUFFER)


def preprocesssvhn(dataset):

  def svhn_batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 32, 32,3]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS_svhn).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(svhn_batch_format_fn).prefetch(PREFETCH_BUFFER)

def preprocessemnist(dataset):

  def emnist_batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1,28, 28,1]),
        y=tf.reshape(element['label'], [-1, 1]))
  return dataset.repeat(NUM_EPOCHS_EMNIST).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(emnist_batch_format_fn).prefetch(PREFETCH_BUFFER)

def preprocessfashion(dataset):

  def fashion_batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1,28, 28,1]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS_fashion).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(fashion_batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data_cifar10(client_data, client_ids):
  return [
      preprocesscifar10(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]


def make_federated_data_fashion(client_data, client_ids):
  return [
      preprocessfashion(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

def make_federated_data_emnist(client_data, client_ids):
  return [
      preprocessemnist(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]
  

def make_federated_data_svhn(client_data, client_ids):
  return [
      preprocesssvhn(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

class Suited():
    def __init__(self, clientId, prefList):
        self.clientId = clientId
        self.prefList = np.array(prefList).T[0].tolist()
        self.suitors = []
        self.assigned = None
    def assign(self,suitor):
        if self.assigned == None:
            suitor.addClient(self)
            self.assigned = suitor
        else:
            self.assigned.removeClient(self)
            suitor.addClient(self)
            self.assigned = suitor

    def addSuitor(self, server):
        if server not in self.suitors:
            self.suitors.append(server)
            return True
        else:
            return False

    def removeSuitor(self, server):
        if server not in self.suitors:
            return False
        else:
            self.suitors.remove(server)
            return True

    def checkPreference(self):
        lowestIndex = 4
        if(len(self.suitors)) ==0:
            return None
        for suitor in self.suitors:
            suitorIndex = self.prefList.index(suitor)
            if (lowestIndex >suitorIndex): lowestIndex = suitorIndex
        return self.prefList[lowestIndex]

class Suitor():
    def __init__(self, clientList):
        self.clientList = []
        for client in clientList:
            self.clientList.append(client[0])
        self.capacity = NUM_CLIENTS_PER_SERVER
        self.suiteds = []
    def addClient(self, suited):
        if suited not in self.suiteds:
            self.suiteds.append(suited)
    def removeClient(self, suited):
        if suited in self.suiteds:
            self.suiteds.remove(suited)
    def notfull(self):
        if (len(self.suiteds)<self.capacity):
            return True 
        else:
            return False
    def getNextPref(self):
        return self.clientList.pop(0)
    def getClients(self):
        clientL = []
        for c in self.suiteds:
            clientL.append(c.clientId)
        if not len(clientL)==NUM_CLIENTS_PER_SERVER:
            print("too long")
        return clientL
    
    
def match(clientList, svhnServer, cifar10Server, fashionServer, emnistServer):
    suiteds = []
    for client in clientList:
        suiteds.append(Suited(client[0], client[1]))
    svhnSuitor= Suitor(svhnServer)    
    cifar10Suitor= Suitor(cifar10Server)    
    fashionSuitor= Suitor(fashionServer)    
    emnistSuitor= Suitor(emnistServer)    

    not_done = True
    while(svhnSuitor.notfull() or cifar10Suitor.notfull() or fashionSuitor.notfull() or emnistSuitor.notfull()):
        roundClients = []
        if svhnSuitor.notfull():
            nextSuited = svhnSuitor.getNextPref()
            client = next((cl for cl in suiteds if cl.clientId == nextSuited),None)
            client.addSuitor("svhn")
            roundClients.append(client)
        if cifar10Suitor.notfull():
            nextSuited = cifar10Suitor.getNextPref()
            client = next((cl for cl in suiteds if cl.clientId == nextSuited),None)
            client.addSuitor("cifar10")
            roundClients.append(client)
        if fashionSuitor.notfull():
            nextSuited = fashionSuitor.getNextPref()
            client = next((cl for cl in suiteds if cl.clientId == nextSuited),None)
            client.addSuitor("fashion")
            roundClients.append(client)
        if emnistSuitor.notfull():
            nextSuited = emnistSuitor.getNextPref()
            client = next((cl for cl in suiteds if cl.clientId == nextSuited),None)
            client.addSuitor("emnist")
            roundClients.append(client)

        for cl in suiteds:
            pref = cl.checkPreference()
            if pref == "svhn":
                
                cl.assign(svhnSuitor)
            if pref == "cifar10":
                cl.assign(cifar10Suitor)
            if pref == "fashion":
                cl.assign(fashionSuitor)
            if pref == "emnist":
                cl.assign(emnistSuitor)
        print(gc.get_count())
        gc.collect()
        print(gc.get_count())

    return svhnSuitor.getClients(), cifar10Suitor.getClients(), fashionSuitor.getClients(), emnistSuitor.getClients()

if __name__=="__main__":
    main()
