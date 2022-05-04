import nest_asyncio
nest_asyncio.apply()
import collections
from csv import writer
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt

NUM_CLIENTS = 10
NUM_EPOCHS_EMNIST = 10
NUM_EPOCHS_celeba = 100
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS = 100

def main():


    np.random.seed(0)
  
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    celeba_train, celeba_test = tff.simulation.datasets.celeba.load_data()
    emnist_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])
    celeba_dataset = celeba_train.create_tf_dataset_for_client(
    celeba_train.client_ids[0])
    global preprocessed_emnist_dataset 
    global preprocessed_celeba_dataset 
    preprocessed_emnist_dataset = preprocessemnist(emnist_dataset)
    preprocessed_celeba_dataset = preprocessceleba(celeba_dataset)
    

    sample_clients_emnist = emnist_train.client_ids[0:NUM_CLIENTS]
    sample_clients_celeba = celeba_train.client_ids[0:NUM_CLIENTS]

    federated_train_data_emnist = make_federated_data_emnist(emnist_train, sample_clients_emnist)
    federated_train_data_celeba = make_federated_data_celeba(celeba_train, sample_clients_celeba)

    celeba_iterative_process = tff.learning.build_federated_averaging_process(
    modelceleba_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=1.0))


    emnist_iterative_process = tff.learning.build_federated_averaging_process(
    modelemnist_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    celebastate = celeba_iterative_process.initialize()
    emniststate  = emnist_iterative_process.initialize()
    celebaacc = []
    emnistacc = []
    for round_num in range(1, NUM_ROUNDS):
        emniststate, emnistmetrics = emnist_iterative_process.next(emniststate, federated_train_data_emnist)
        celebastate, celebametrics = celeba_iterative_process.next(celebastate, federated_train_data_celeba)
        print('round {:2d}, emnistmetrics={}'.format(round_num, emnistmetrics))
        print('round {:2d}, celebametrics={}'.format(round_num, celebametrics))
        print(celebametrics['train']['sparse_categorical_accuracy'])
        celebaacc.append([float(celebametrics['train']['sparse_categorical_accuracy'])])
        emnistacc.append([float(emnistmetrics['train']['sparse_categorical_accuracy'])])
        print(celebaacc)
    with open('celeba.csv','a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerows(celebaacc)
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

def create_keras_celeba_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(84,84,3), batch_size= BATCH_SIZE),
      
      tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation = 'relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation = 'relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
      tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation = 'relu'),
      tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, kernel_initializer='zeros'),
      tf.keras.layers.Dense(256, kernel_initializer='zeros'),
      tf.keras.layers.Dense(128, kernel_initializer='zeros'),
      tf.keras.layers.ReLU(),
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

def modelceleba_fn():
# We _must_ create a new model here, and _not_ capture it from an external
# scope. TFF will call this within different graph contexts.
    
    keras_model = create_keras_celeba_model()
    return tff.learning.from_keras_model(
    keras_model,
    input_spec=preprocessed_celeba_dataset.element_spec,
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

def preprocessceleba(dataset):

  def celeba_batch_format_fn(element):
    
    return collections.OrderedDict(
        x=tf.reshape(element['image'],[BATCH_SIZE, 84,84,3]),
        y=tf.reshape(element['label'], [-1, 40]))

  return dataset.repeat(NUM_EPOCHS_celeba).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(celeba_batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data_celeba(client_data, client_ids):
  return [
      preprocessceleba(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

def make_federated_data_emnist(client_data, client_ids):
  return [
      preprocessemnist(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

if __name__=="__main__":
    main()