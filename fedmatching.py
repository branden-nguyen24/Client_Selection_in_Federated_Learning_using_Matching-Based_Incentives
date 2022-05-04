import nest_asyncio
nest_asyncio.apply()
import collections
from csv import writer
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt

NUM_CLIENTS = 50
NUM_EPOCHS_EMNIST = 100
NUM_EPOCHS_fashion = 100
BATCH_SIZE = 30
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS =1000

def main():
    

    split = 100

    np.random.seed(0)
    
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
   
    (x_train, y_train), (x_test, y_test)  = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int32)
    x_test = x_test.astype(np.float32).reshape(10000, 28, 28, 1)
    y_test = y_test.astype(np.int32).reshape(10000, 1)

    total_image_count = len(x_train)
    image_per_set = int(np.floor(total_image_count/split))

    client_train_dataset = collections.OrderedDict()
    for i in range(1, split+1):
        client_name = "client_" + str(i)
        start = image_per_set * (i-1)
        end = image_per_set * i

        print(f"Adding data from {start} to {end} for client : {client_name}")
        data = collections.OrderedDict((('label', y_train[start:end]), ('pixels', x_train[start:end])))
        client_train_dataset[client_name] = data
                        
    fashion_data = tff.simulation.datasets.TestClientData(client_train_dataset)
    fashion_dataset = fashion_data.create_tf_dataset_for_client(fashion_data.client_ids[0])
    emnist_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])
  
    global preprocessed_emnist_dataset 
    global preprocessed_fashion_dataset 
    preprocessed_emnist_dataset = preprocessemnist(emnist_dataset)
    preprocessed_fashion_dataset = preprocessfashion(fashion_dataset)
    

    sample_clients_emnist = emnist_train.client_ids[0:NUM_CLIENTS]
    sample_clients_fashion = fashion_data.client_ids[0:NUM_CLIENTS]

    federated_train_data_emnist = make_federated_data_emnist(emnist_train, sample_clients_emnist)
    federated_train_data_fashion = make_federated_data_fashion(fashion_data, sample_clients_fashion)

    fashion_iterative_process = tff.learning.build_federated_averaging_process(
    modelfashion_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.002),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))


    emnist_iterative_process = tff.learning.build_federated_averaging_process(
    modelemnist_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.002),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))

    fashionstate = fashion_iterative_process.initialize()
    emniststate  = emnist_iterative_process.initialize()
    fashionacc = []
    emnistacc = []
    for round_num in range(1, NUM_ROUNDS):
        emniststate, emnistmetrics = emnist_iterative_process.next(emniststate, federated_train_data_emnist)
        fashionstate, fashionmetrics = fashion_iterative_process.next(fashionstate, federated_train_data_fashion)
        print('round {:2d}, emnistmetrics={}'.format(round_num, emnistmetrics))
        print('round {:2d}, fashionmetrics={}'.format(round_num, fashionmetrics))
        print(fashionmetrics['train']['sparse_categorical_accuracy'])
        fashionacc.append([float(fashionmetrics['train']['sparse_categorical_accuracy'])])
        emnistacc.append([float(emnistmetrics['train']['sparse_categorical_accuracy'])])
        print(fashionacc)
    with open('fashion.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerows(fashionacc)
        f_object.close()
    with open('emnist.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerows(emnistacc)
        f_object.close()
def create_keras_emnist_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])

def create_keras_fashion_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(784,)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(256, kernel_regularizer='l1_l2'),
      tf.keras.layers.Dense(128, kernel_regularizer='l1_l2'),
      tf.keras.layers.Dense(10),
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
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS_fashion).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(fashion_batch_format_fn).prefetch(PREFETCH_BUFFER)


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

if __name__=="__main__":
    main()