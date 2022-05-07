from typing import Dict, OrderedDict
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
NUM_ROUNDS =1000
VARIANCE = 0.2
randomFEDAVG = False
def main():
    

    split = 600

    np.random.seed(0)

    svhn_client_train_dataset = collections.OrderedDict()

    svhn_train=tfds.load("svhn_cropped", split= 'train')
    svhn_test=tfds.load("svhn_cropped", split= 'test')
    svhn_client_train_dataset = collections.OrderedDict()
    svhn_train_x=[]
    svhn_train_y=[]
    for element in svhn_train.take(-1):
        svhn_train_x.append(tfds.as_numpy(element['image']).astype(np.float32))
        svhn_train_y.append(tfds.as_numpy(element['label']).astype(np.int32))

    svhn_total_image_count = len(svhn_train_x)
    svhn_image_per_set = int(np.floor(svhn_total_image_count/split))


    end =-1
    for i in range(1, split+1):
        client_name = "client_" + str(i)
        start = end + 1
        end = end + svhn_image_per_set+round(svhn_image_per_set*(2*(np.random.rand()-0.5)*VARIANCE))
        if(end>len(svhn_train)-1): end = len(svhn_train_x) - 1
        print(f"Adding data from {start} to {end} for client : {client_name}")
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

    


    cifar10_data = tff.simulation.datasets.TestClientData(cifar10_client_train_dataset)                 
    fashion_data = tff.simulation.datasets.TestClientData(fashion_client_train_dataset)
    svhn_data = tff.simulation.datasets.TestClientData(svhn_client_train_dataset)
    emnist_data = tff.simulation.datasets.TestClientData(emnist_client_train_dataset)
    
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
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.002),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.5))

    fashion_iterative_process = tff.learning.build_federated_averaging_process(
    modelfashion_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.001),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))


    emnist_iterative_process = tff.learning.build_federated_averaging_process(
    modelemnist_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.5))

    svhn_iterative_process = tff.learning.build_federated_averaging_process(
    modelsvhn_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.002),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

    
    
    
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
        
        if(randomFEDAVG==False):
            
            
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


        federated_train_data_svhn = make_federated_data_svhn(svhn_data, round_client_list_svhn)
        federated_train_data_emnist = make_federated_data_emnist(emnist_data, round_client_list_emnist)
        federated_train_data_fashion = make_federated_data_fashion(fashion_data, round_client_list_fashion)
        federated_train_data_cifar10 = make_federated_data_cifar10(cifar10_data, round_client_list_cifar10)

        

        cifar10state, cifar10metrics = cifar10_iterative_process.next(cifar10state, federated_train_data_cifar10)
        emniststate, emnistmetrics = emnist_iterative_process.next(emniststate, federated_train_data_emnist)
        fashionstate, fashionmetrics = fashion_iterative_process.next(fashionstate, federated_train_data_fashion)
        svhnstate, svhnmetrics = svhn_iterative_process.next(svhnstate, federated_train_data_svhn)

        print('round {:2d}, emnistmetrics={}'.format(round_num, emnistmetrics))
        print('round {:2d}, fashionmetrics={}'.format(round_num, fashionmetrics))
        print('round {:2d}, svhnmetrics={}'.format(round_num, svhnmetrics))
        print('round {:2d}, svhnmetrics={}'.format(round_num, cifar10metrics))

        fashionacc.append([float(fashionmetrics['train']['sparse_categorical_accuracy'])])
        emnistacc.append([float(emnistmetrics['train']['sparse_categorical_accuracy'])])
        svhnacc.append([float(svhnmetrics['train']['sparse_categorical_accuracy'])])
        cifar10acc.append([float(cifar10metrics['train']['sparse_categorical_accuracy'])])
        
    with open('fashion.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerows(fashionacc)
        f_object.close()
    with open('emnist.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerows(emnistacc)
        f_object.close()
    with open('svhn.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerows(svhnacc)
        f_object.close()
    with open('cifar10.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerows(cifar10acc)
        f_object.close()
def create_keras_emnist_model():
  return tf.keras.models.Sequential([
      
      tf.keras.layers.InputLayer(input_shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
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
        x=tf.reshape(element['pixels'], [-1, 784]),
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

roroundund

if __name__=="__main__":
    main()