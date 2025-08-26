import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import time
import numpy as np
import flax.linen as nn # flax.linen for nn.Module and layers
import other 
import metrics as met
import jax


class CnnClassifier(nn.Module):
  num_classes: int # Number of output classes (e.g., 10 for digits)

  @nn.compact
  def __call__(self, x):
  
    input_is_single_image = False
    if len(x.shape) == 1:
        # Single image input (e.g., from vmap)
        input_is_single_image = True
        x = x.reshape((1, 28, 28, 1))
    elif len(x.shape) == 2:
        # Batched input (e.g., from init, or if you were to call apply directly with a batch)
        batch_size = x.shape[0]
        x = x.reshape((batch_size, 28, 28, 1))
    else:
        raise ValueError("Unexpected input shape for CnnClassifier: {}".format(x.shape))

    # Convolutional Layer 1
    # features: number of output channels (filters)
    # kernel_size: (height, width) of the convolution window
    x = nn.Conv(features=32, kernel_size=(3, 3), name='conv1')(x)
    x = nn.relu(x)

    # Max Pooling Layer 1
    # window_shape: (height, width) of the pooling window
    # strides: (height, width) of the stride
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    
    x = nn.Conv(features=64, kernel_size=(3, 3), name='conv2')(x)
    x = nn.relu(x)

    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = x.reshape((x.shape[0], -1)) # -1 infers the dimension based on the other dimensions

    x = nn.Dense(features=128, name='fc1')(x)
    x = nn.relu(x)

    # Output Dense Layer
    logits = nn.Dense(features=self.num_classes, name='logits')(x)
    
    if input_is_single_image:
        return logits.squeeze(axis=0) # Squeeze the batch dimension, making it (num_classes,)
    else:
        return logits # For batches, keep as (batch_size, num_classes)


def init_network_params(input_shape, key, num_classes):
  """
  Initializes the parameters for the CNN classifier.
  
  Args:
    input_shape: The shape of a single input image (e.g., (784,)).
    key: JAX PRNG key for initialization.
    num_classes: The number of output classes.
  
  Returns:
    The initialized parameters for the CnnClassifier.
  """
  cnn_model = CnnClassifier(num_classes=num_classes)
  
  # Initialize the parameters 
  dummy_input = jnp.ones((1,) + input_shape, jnp.float32)
  
  params = cnn_model.init(key, dummy_input)['params']
  return params

# The predict function 
# It will receive the model parameters and the input image (flattened)
def predict(params, image):
  cnn_model = CnnClassifier(num_classes=params['logits']['kernel'].shape[1]) # Infer num_classes from output layer
  
  # Apply the model with the given parameters
  logits = cnn_model.apply({'params': params}, image)
  return logits - logsumexp(logits) # Apply logsumexp for numerical stability


# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

def loss(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)

def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  #print(f"Shape of batched_predict: {batched_predict(params, images).shape}")
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  # Add these print statements for debugging
  #print(f"Shape of predicted_class: {predicted_class.shape}")
  #print(f"Shape of target_class: {target_class.shape}")
  return jnp.mean(predicted_class == target_class)

@jit
def update(params, x, y, step_size):
  grads = grad(loss)(params, x, y)
  # Correctly update Flax parameters using tree_map
  return jax.tree_util.tree_map(lambda p, g: p - step_size * g, params, grads)

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def batch_generator(data,data_labels, batch_size):
    num_samples = data.shape[0]
    indices = jnp.arange(num_samples)
    
    while True:
        # Shuffle indices for each epoch
        shuffled_indices = random.permutation(random.PRNGKey(1000), indices) 
        
        for i in range(0, num_samples, batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]
            yield data[batch_indices],data_labels[batch_indices]


def class_results_combine(train_ds1, train_ds2, train_labs, 
                          original_top_view_test, original_bottom_view_test, # Original test views
                          processed_top_view_test, processed_bottom_view_test, # VAE-processed test views (either original or generated)
                          test_labs,
                          full_train1, full_train2, full_train_labs, 
                          params, num_epochs=30, batch_num=32, n_targets=10, step_size=0.01, start_range=0, end_range=784):
  """
  Trains a classifier and evaluates it on various combinations of original and VAE-processed image halves,
  ensuring spatial consistency for top and bottom views.

  Args:
    train_ds1: Top halves of training images.
    train_ds2: Bottom halves of training images.
    train_labs: Labels for training images.
    original_top_view_test: Original top halves of test images.
    original_bottom_view_test: Original bottom halves of test images.
    processed_top_view_test: The top half of the test images, which is either the original
                             top half (if it was the input view) or the generated top half
                             (if it was the masked view).
    processed_bottom_view_test: The bottom half of the test images, which is either the original
                                bottom half (if it was the input view) or the generated bottom half
                                (if it was the masked view).
    test_labs: Labels for test images.
    full_train1: Full training set top halves (for full train accuracy calculation).
    full_train2: Full training set bottom halves (for full train accuracy calculation).
    full_train_labs: Labels for full training set.
    params: Initial classifier parameters.
    num_epochs: Number of training epochs for the classifier.
    batch_num: Batch size for training.
    n_targets: Number of target classes (e.g., 10 for digits).
    step_size: Learning rate for the classifier.
    start_range: Start index for image features (usually 0).
    end_range: End index for image features (usually 784 for full image).

  Returns:
    train_accuracies: List of training accuracies per epoch.
    test_accuracies: List of test accuracies on full original images per epoch.
    recon_accuracies1: List of test accuracies on (Original Top + Generated Bottom) images per epoch.
    recon_accuracies2: List of test accuracies on (Generated Top + Original Bottom) images per epoch.
    recon_accuracies_full_processed: List of test accuracies on (Processed Top + Processed Bottom) images per epoch.
  """
  train_accuracies = []
  test_accuracies = []
  recon_accuracies1 = [] # For Original Top + Generated Bottom
  recon_accuracies2 = [] # For Generated Top + Original Bottom
  recon_accuracies_full_processed = [] # For Processed Top + Processed Bottom (the main one you want)

  # Prepare full original training and test sets
  train_ds_full_original = jnp.concatenate([train_ds1, train_ds2], axis=1)
  test_ds_full_original = jnp.concatenate([original_top_view_test, original_bottom_view_test], axis=1)
  full_train_full_original = jnp.concatenate([full_train1, full_train2], axis=1)

  # 1. Test set: Original Top View + Generated Bottom View
  # This combines the original top half with the VAE-generated bottom half.
  test_ds_orig_top_gen_bottom = jnp.concatenate([original_top_view_test, processed_bottom_view_test], axis=1)

  # 2. Test set: Generated Top View + Original Bottom View
  # This combines the VAE-generated top half with the original bottom half.
  test_ds_gen_top_orig_bottom = jnp.concatenate([processed_top_view_test, original_bottom_view_test], axis=1)

  # 3. Test set: Processed Top View + Processed Bottom View
  # This is the primary set you're interested in: the full digit formed by the
  # spatially correct processed (original or generated) halves.
  test_ds_full_processed = jnp.concatenate([processed_top_view_test, processed_bottom_view_test], axis=1)

  train_ds_iterator = batch_generator(train_ds_full_original,train_labs, batch_num)
  #train_labs_batches = batch_generator(train_labs, batch_num)

  for epoch in range(num_epochs):
    # Training loop for the classifier
    for _ in range(len(full_train_full_original) // batch_num):
      #x = next(train_ds_batches)[:,start_range:end_range]
      #y = next(train_labs_batches)
      x,y = next(train_ds_iterator)
      y = one_hot(y, n_targets)
      params = update(params, x, y, step_size)

    # Calculate accuracies for various test sets
    train_acc = accuracy(params, full_train_full_original[:,start_range:end_range], one_hot(full_train_labs, n_targets))
    train_accuracies.append(train_acc)
    
  # Accuracy on the full original test images
  test_acc = accuracy(params, test_ds_full_original[:,start_range:end_range], one_hot(test_labs, n_targets))
    
  # Accuracy on (Original Top + Generated Bottom)
  recon_test_acc1 = accuracy(params, test_ds_orig_top_gen_bottom[:,start_range:end_range], one_hot(test_labs, n_targets))
    
  # Accuracy on (Generated Top + Original Bottom)
  recon_test_acc2 = accuracy(params, test_ds_gen_top_orig_bottom[:,start_range:end_range], one_hot(test_labs, n_targets))
    
  # Accuracy on (Processed Top + Processed Bottom) - This is the main one you wanted
  recon_test_acc_full_processed = accuracy(params, test_ds_full_processed[:,start_range:end_range], one_hot(test_labs, n_targets))
    
  test_accuracies.append(test_acc)
  recon_accuracies1.append(recon_test_acc1)
  recon_accuracies2.append(recon_test_acc2)
  recon_accuracies_full_processed.append(recon_test_acc_full_processed)

  return train_accuracies, test_accuracies, recon_accuracies1, recon_accuracies2, recon_accuracies_full_processed


def class_results(train_ds, train_labs, test_ds, test_labs,
                   full_train, full_train_labs, params, num_epochs=30, batch_num=32, n_targets=10, step_size=0.01, start_range=0, end_range=784, cut=0.2):
  # This function trains and evaluates a classifier on a single "view". It's primarily used to establish a baseline
  # or to understand how well a digit can be classified using only one half (e.g., top or bottom) of the original image data.
  train_accuracies = []
  test_accuracies = []
  for epoch in range(num_epochs):
    
    for _ in range(len((full_train))//batch_num):
        x = next(train_ds)[:,start_range:end_range]
        y = next(train_labs)
        y = one_hot(y, n_targets)
        params = update(params, x, y, step_size)
    
    train_acc = accuracy(params, full_train[:,start_range:end_range], one_hot(full_train_labs, n_targets))
    train_accuracies.append(train_acc)
  
  test_acc = accuracy(params, test_ds[:,start_range:end_range], one_hot(test_labs, n_targets))   
  test_accuracies.append(test_acc)
  
  return train_accuracies, test_accuracies

# To call it in the main script
#import classification_modified as cla
#import jax.random as random

# Assuming input_dim is 784 (28*28) and n_targets is 10
#input_dim = 784 
#n_targets = 10
#key_for_classifier_init = random.key(123) # Use a specific seed for reproducibility

# Initialize the CNN classifier parameters
#params_full = cla.init_network_params(input_shape=(input_dim,), key=key_for_classifier_init, num_classes=n_targets)
