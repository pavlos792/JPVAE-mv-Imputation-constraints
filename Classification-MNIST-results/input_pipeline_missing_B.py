# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input pipeline for VAE dataset."""

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


def build_train_val_set(batch_size, ds_builder,key, percentage_to_keep,validation_split_percentage=10, total_training_samples=50000):
  """Builds train dataset."""

  train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
  labs = train_ds.map(extract_labs)
  train_ds = train_ds.map(prepare_image)
  
  train_ds = train_ds.shuffle(total_training_samples, seed=key)
  labs = labs.shuffle(total_training_samples, seed=key)

  # Calculate the number of samples to keep using the new input parameter.
  # Convert the percentage (e.g., 90) to a fraction (0.90).
  num_samples_to_keep = int(total_training_samples * (percentage_to_keep / 100.0))

  train_ds = train_ds.take(num_samples_to_keep)
  labs = labs.take(num_samples_to_keep)
  # Calculate split sizes
  num_val_samples = int(num_samples_to_keep * (validation_split_percentage / 100.0))
  num_train_samples = num_samples_to_keep - num_val_samples
  
  # Split into training and validation datasets
  train_ds_actual = train_ds.skip(num_val_samples) # Skip validation samples for training
  train_labs_actual = labs.skip(num_val_samples)
  val_ds_actual = train_ds.take(num_val_samples)   # Take validation samples
  val_labs_actual = labs.take(num_val_samples)
  
  train_ds_batched = train_ds_actual.repeat().batch(batch_size)
  train_labs_batched = train_labs_actual.repeat().batch(batch_size)
  train_ds_iter = iter(tfds.as_numpy(train_ds_batched))
  train_labs_iter = iter(tfds.as_numpy(train_labs_batched))
  
  val_ds_batched = val_ds_actual.repeat().batch(batch_size)
  val_labs_batched = val_labs_actual.repeat().batch(batch_size)
  val_ds_iter = iter(tfds.as_numpy(val_ds_batched))
  val_labs_iter = iter(tfds.as_numpy(val_labs_batched))
  
  return train_ds_iter, train_labs_iter, val_ds_iter, val_labs_iter

def build_full_train(ds_builder, key, percentage_to_keep,total_training_samples=50000):
  """Builds train dataset."""
  test_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
  test_ds = test_ds.shuffle(total_training_samples, seed=key)
  num_samples_to_keep = int(total_training_samples * (percentage_to_keep / 100.0))
  limited_dataset = test_ds.take(num_samples_to_keep)
  labs = jnp.array(list(limited_dataset.map(extract_labs).batch(num_samples_to_keep))[0])
  images = limited_dataset.map(prepare_image).batch(num_samples_to_keep)
  images_jnp = jnp.array(list(images)[0])
  
  images_jnp = jax.device_put(images_jnp)
  return images_jnp, labs
  
def build_full_train_with_val(ds_builder, key, percentage_to_keep, validation_split_percentage=10, total_training_samples=50000): 
  """
  Builds the full training dataset (or a subset based on percentage_to_keep)
  as a single JAX array. This data is intended for training the classifier.
  It is the same subset of data that the VAE was trained on (train_ds_actual from build_train_val_set).
  """
  full_train_dataset_tf = ds_builder.as_dataset(split=tfds.Split.TRAIN)

  full_train_dataset_tf = full_train_dataset_tf.shuffle(total_training_samples, seed=key)

  total_samples_for_train_val = int(total_training_samples * (percentage_to_keep / 100.0))
  
  # Take the specified number of samples from the shuffled dataset
  train_val_ds_subset = full_train_dataset_tf.take(total_samples_for_train_val)

  # Calculate split sizes for actual train and validation
  num_val_samples = int(total_samples_for_train_val * (validation_split_percentage / 100.0))
  num_train_samples = total_samples_for_train_val - num_val_samples
  
  mapped_ds_subset = train_val_ds_subset.map(lambda x: (prepare_image(x), extract_labs(x)))
  
  # Split into training and validation datasets
  train_ds_actual_tf = mapped_ds_subset.skip(num_val_samples)
  val_ds_actual_tf = mapped_ds_subset.take(num_val_samples)
  
  full_train_data_batch = list(train_ds_actual_tf.batch(num_train_samples))[0]
  full_train_images_jnp = jnp.array(full_train_data_batch[0])
  full_train_labs_jnp = jnp.array(full_train_data_batch[1])

  full_val_data_batch = list(val_ds_actual_tf.batch(num_val_samples))[0]
  full_val_images_jnp = jnp.array(full_val_data_batch[0])
  full_val_labs_jnp = jnp.array(full_val_data_batch[1])

  full_train_images_jnp = jax.device_put(full_train_images_jnp)
  full_train_labs_jnp = jax.device_put(full_train_labs_jnp)
  full_val_images_jnp = jax.device_put(full_val_images_jnp)
  full_val_labs_jnp = jax.device_put(full_val_labs_jnp)

  return full_train_images_jnp, full_train_labs_jnp, full_val_images_jnp, full_val_labs_jnp


def build_test_set(ds_builder,TOTAL_OFFICIAL_TEST=10000):
  """Builds train dataset."""
  test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
  
  mapped_dataset = test_ds.map(lambda x: (prepare_image(x), extract_labs(x)))
  
  full_data_batch = list(mapped_dataset.batch(TOTAL_OFFICIAL_TEST))[0]
  
  images_jnp = jnp.array(full_data_batch[0])
  labs_jnp = jnp.array(jnp.array(full_data_batch[1])) 
  
  images_test_jnp = jax.device_put(images_jnp)
  labs_test_jnp = jax.device_put(labs_jnp)
  return images_test_jnp, labs_test_jnp
  
def build_complementary_test_set(ds_builder, key, percentage_to_keep,total_training_samples=50000):
  """Builds the complementary dataset, all the samples that are not used in the training set."""
  test_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
  test_ds = test_ds.shuffle(total_training_samples, seed=key)
  num_samples_to_keep = int(total_training_samples * (percentage_to_keep / 100.0))
  
  complementary_dataset = test_ds.skip(num_samples_to_keep)
  complementary_samples = total_training_samples - num_samples_to_keep
  
  mapped_dataset = complementary_dataset.map(lambda x: (prepare_image(x), extract_labs(x)))
  
  full_data_batch = list(mapped_dataset.batch(complementary_samples))[0]
  images_jnp = jnp.array(full_data_batch[0])
  # Ensure labels are 2D (1, N) by expanding dimensions
  #labs_jnp = jnp.expand_dims(jnp.array(full_data_batch[1]), axis=0)
  labs_jnp = jnp.array(jnp.array(full_data_batch[1]))
  
  images_jnp = jax.device_put(images_jnp)
  labs_jnp = jax.device_put(labs_jnp)
  
  return images_jnp, labs_jnp


def build_combined_test_set(ds_builder,key, percentage_to_keep):
  """Builds combined test dataset dataset."""
  
  if percentage_to_keep == 100:
    # official test set
    combined_images, combined_labs = build_test_set(ds_builder)
    
  else:
    # complementary test set
    comp_test_images, comp_test_labs = build_complementary_test_set(ds_builder, key, percentage_to_keep)
  
    # official test set
    official_test_images, official_test_labs = build_test_set(ds_builder)
  
    # Concatenate the images and labels
    combined_images = jnp.concatenate([comp_test_images, official_test_images], axis=0)
    combined_labs = jnp.concatenate([comp_test_labs, official_test_labs], axis=0)
    
  
  return combined_images, combined_labs


def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)/255.0
  x = tf.reshape(x, (-1,))
  return x

def extract_labs(x):
  return x['label']