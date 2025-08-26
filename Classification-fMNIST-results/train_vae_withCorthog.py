import os
import jax
import scipy.ndimage
import jax.numpy as jnp
import vae_orthog as vae
import metrics as met
from jax import random
from jax import vmap
from flax import linen as nn
import other
from flax.training import train_state
import optax
import utils as vae_utils
import ml_collections
import training_steps as ts 

def eval(params, images1, images2, z_1, z_2, z_rng, latents, num_out, alpha, binary,param, zero=False, print_images=False, h=14, w=28, results_dir=None, figures_dir=None):
  rngs = random.split(z_rng, 50)
  n_images = images1.shape[0]
  def eval_model(vae):
    loss = []
    for i in range(50):
      recon_images1, mean1, logvar1, z1, recon_images2, mean2, logvar2, z2, mat = vae(images1, images2, rngs[i])
      matrices = other.make_matrices(latents, mat)
      if zero:
        metrics = met.compute_metrics_zero(recon_images1, recon_images2, images1, images2,
                                  mean1, logvar1, mean2, logvar2, False)
      elif binary:
        metrics = met.compute_metrics_bin_test(recon_images1, recon_images2, images1, images2,
                                  mean1, logvar1, mean2, logvar2, matrices, False)
      else:
        metrics = met.compute_metrics(recon_images1, recon_images2, images1, images2,
                                  mean1, logvar1, mean2, logvar2, matrices, False)
      loss.append(metrics['loss'])

    ll = jnp.mean(nn.activation.logsumexp(-jnp.asarray(loss),axis=0) - jnp.ones(n_images)*jnp.log(n_images/20))
    recon_images1, mean1, logvar1, z1, recon_images2, mean2, logvar2, z2, mat = vae(images1, images2, rngs[0])
    if zero:
      metrics = met.compute_metrics_zero(recon_images1, recon_images2, images1, images2, mean1, logvar1, mean2, logvar2, other.make_matrices(latents, mat))
    elif binary:
      metrics = met.compute_metrics_bin_test(recon_images1, recon_images2, images1, images2, mean1, logvar1, mean2, logvar2, other.make_matrices(latents, mat))
    else:
      metrics = met.compute_metrics(recon_images1, recon_images2, images1, images2, mean1, logvar1, mean2, logvar2, other.make_matrices(latents, mat))
    if print_images:
      comparison = jnp.concatenate([
          images1[:8].reshape(-1, h, w, 1),
          images2[:8].reshape(-1, h, w, 1),
          recon_images1[:8].reshape(-1, h, w, 1),
          recon_images2[:8].reshape(-1, h, w, 1),
          images1[8:16].reshape(-1, h, w, 1),
          images2[8:16].reshape(-1, h, w, 1),
          recon_images1[8:16].reshape(-1, h, w, 1),
          recon_images2[8:16].reshape(-1, h, w, 1),
          images1[16:24].reshape(-1, h, w, 1),
          images2[16:24].reshape(-1, h, w, 1),
          recon_images1[16:24].reshape(-1, h, w, 1),
          recon_images2[16:24].reshape(-1, h, w, 1),
          images1[24:32].reshape(-1, h, w, 1),
          images2[24:32].reshape(-1, h, w, 1),
          recon_images1[24:32].reshape(-1, h, w, 1),
          recon_images2[24:32].reshape(-1, h, w, 1)
      ])
      generate_images1, generate_images2 = vae.generate(z_1, z_2)
      generate_images1 = generate_images1.reshape(-1, h, w, 1)
      generate_images2 = generate_images2.reshape(-1, h, w, 1)
    else:
      comparison = jnp.zeros((1,1))
      generate_images1 = jnp.zeros((1,1))
      generate_images2 = jnp.zeros((1,1))
    
    return metrics, comparison, generate_images1, generate_images2, ll, z1, z2, mat
  return nn.apply(eval_model, vae.model(latents, num_out, alpha))({'params': params})
eval_f = jax.jit(eval, static_argnums=(6,7, 8, 9, 10,11,12,15,16))

def eval_generation(params, images1, images2, z_1, z_2, main_rng, latents, num_out, alpha, binary,param, zero=False, print_images=False, h=14, w=28, results_dir=None, figures_dir=None,mask_rng=None):
  
  z_rng, mask_rng_local = random.split(main_rng) # Split the incoming main_rng

  rngs = random.split(z_rng, 50) # Use z_rng for Monte Carlo sampling of z
  n_images = images1.shape[0]

  # Generate random mask choices using the newly split mask_rng_local
  mask_choices = random.bernoulli(mask_rng_local, p=0.5, shape=(n_images,))
  
  def eval_model(vae):
    loss = []
    all_recon_input_views = []
    all_generated_missing_views = []
    all_original_input_views_mc = []
    all_original_missing_views_mc = []
    
    for i in range(50):
      # Prepare inputs for VAE based on mask_choices for the current batch
      # If mask_choices[k] is 0, input_x1_masked[k] = images1[k], input_x2_masked[k] = zeros
      # If mask_choices[k] is 1, input_x1_masked[k] = zeros, input_x2_masked[k] = images2[k]
      input_x1_masked = jnp.where(mask_choices[:, None], jnp.zeros_like(images1), images1)
      input_x2_masked = jnp.where(mask_choices[:, None], images2, jnp.zeros_like(images2))

      # VAE forward pass with masked inputs to get latents and matrices
      # For vae_evals, this returns U, S, V directly from the apply method.
      
      recon_x1_dummy, mean1, logvar1, z1, recon_x2_dummy, mean2, logvar2, z2, mat_for_metrics = vae.apply(
          {'params': params}, input_x1_masked, input_x2_masked, rngs[i]
      )
      
      if zero:
         # Independence case (C = 0):
         # Decode the zero latent for the missing view.
         # If view 1 is missing -> generate it from z1 = 0 via decoder1
         # If view 2 is missing -> generate it from z2 = 0 via decoder2
         zero_z1 = jnp.zeros_like(z1)
         zero_z2 = jnp.zeros_like(z2)

         #recon_x1_zero = vae.apply({'params': params}, zero_z1,  method=lambda m, z: m.decoder1(z, apply_sigmoid=True))
         #recon_x2_zero = vae.apply({'params': params}, zero_z2,  method=lambda m, z: m.decoder2(z, apply_sigmoid=True))
         recon_x1_zero = vae.decoder1(zero_z1, apply_sigmoid=True)
         recon_x2_zero = vae.decoder2(zero_z2, apply_sigmoid=True)
         
         recon_x1_cond_gen = recon_x1_zero
         recon_x2_cond_gen = recon_x2_zero


      else:# Conditional generation using vae_model_instance.cross_gen
          # It takes the latent codes (z1, z2) and returns the conditionally generated views.
          # We pass the actual z1 and z2 from the masked input, and cross_gen handles the prediction.
         z1_cond_gen, recon_x1_cond_gen, z2_cond_gen, recon_x2_cond_gen = vae.cross_gen(z1, z2, rngs[i])
      
      # Determine which view was provided and which was generated for this sample
      # recon_input_view_current: reconstruction of the view that was *provided*
      # generated_missing_view_current: the view that was *generated*
      recon_input_view_current = jnp.where(mask_choices[:, None], recon_x2_dummy, recon_x1_dummy) # dummy here refers to the direct output from apply
      generated_missing_view_current = jnp.where(mask_choices[:, None], recon_x1_cond_gen, recon_x2_cond_gen)
      
      # Get the mean/logvar of the *provided* view for metrics
      mean_provided = jnp.where(mask_choices[:, None], mean2, mean1)
      logvar_provided = jnp.where(mask_choices[:, None], logvar2, logvar1)

      # Get the original ground truth for the provided and missing views for metrics
      original_input_view_current = jnp.where(mask_choices[:, None], images2, images1)
      original_missing_view_current = jnp.where(mask_choices[:, None], images1, images2)
      
      # For metrics, we still need the matrices from the VAE's apply pass
      matrices = other.make_matrices(latents, mat_for_metrics)
      
      # one view is intentionally masked (set to zeros) and thus its latent statistics are not meaningfully encoded from data, thus jnp.zeros_like(mean) and (logvar).
      if zero:
        metrics_iter = met.compute_metrics_zero(recon_input_view_current, generated_missing_view_current, original_input_view_current, original_missing_view_current,
                                  mean_provided, logvar_provided, jnp.zeros_like(mean_provided), jnp.zeros_like(logvar_provided), False)
      elif binary:
        metrics_iter = met.compute_metrics_bin_test(recon_input_view_current, generated_missing_view_current, original_input_view_current, original_missing_view_current,
                                  mean_provided, logvar_provided, jnp.zeros_like(mean_provided), jnp.zeros_like(logvar_provided), matrices, False)
      else:
        metrics_iter = met.compute_metrics(recon_input_view_current, generated_missing_view_current, original_input_view_current, original_missing_view_current,
                                  mean_provided, logvar_provided, jnp.zeros_like(mean_provided), jnp.zeros_like(logvar_provided), matrices)
          
      loss.append(metrics_iter['loss'])
      all_recon_input_views.append(recon_input_view_current)
      all_generated_missing_views.append(generated_missing_view_current)
      all_original_input_views_mc.append(original_input_view_current)
      all_original_missing_views_mc.append(original_missing_view_current)

    
    ll = jnp.mean(nn.activation.logsumexp(-jnp.asarray(loss),axis=0) - jnp.ones(n_images)*jnp.log(n_images/20))
    
    ## Average the reconstructed/generated views and original views over the 50 runs
    final_recon_input_views = jnp.mean(jnp.stack(all_recon_input_views), axis=0)
    final_generated_missing_views = jnp.mean(jnp.stack(all_generated_missing_views), axis=0)
    final_original_input_views = jnp.mean(jnp.stack(all_original_input_views_mc), axis=0)
    final_original_missing_views = jnp.mean(jnp.stack(all_original_missing_views_mc), axis=0)
    
    top_half_processed = jnp.where(mask_choices[:, None], final_generated_missing_views, final_recon_input_views)
    bottom_half_processed = jnp.where(mask_choices[:, None], final_recon_input_views, final_generated_missing_views)
    top_half_withoriginal = jnp.where(mask_choices[:, None], final_generated_missing_views, final_original_input_views)
    bottom_half_withoriginal = jnp.where(mask_choices[:, None], final_original_input_views, final_generated_missing_views)

    
    # Get final_mat from a single forward pass for consistency (using first rng)
    # Use the masked inputs for this pass as well.
    _, _, _, _, _, _, _, _, final_mat = vae.apply({'params': params}, input_x1_masked, input_x2_masked, rngs[0])
    
    matrices = other.make_matrices(latents, final_mat)
    
    if zero:
      metrics_iter = met.compute_metrics_zero(final_recon_input_views, final_generated_missing_views, final_original_input_views, final_original_missing_views,
                                  mean_provided, logvar_provided, jnp.zeros_like(mean_provided), jnp.zeros_like(logvar_provided), False)
    elif binary:
      metrics_iter = met.compute_metrics_bin_test(final_recon_input_views, final_generated_missing_views, final_original_input_views, final_original_missing_views,
                                  mean_provided, logvar_provided, jnp.zeros_like(mean_provided), jnp.zeros_like(logvar_provided), matrices, False)
    else:
      metrics_iter = met.compute_metrics(final_recon_input_views, final_generated_missing_views, final_original_input_views, final_original_missing_views,
                                  mean_provided, logvar_provided, jnp.zeros_like(mean_provided), jnp.zeros_like(logvar_provided), matrices)
    
    if print_images:
      
      #top_half_processed = jnp.where(mask_choices[:, None], final_generated_missing_views, final_recon_input_views)
      #bottom_half_processed = jnp.where(mask_choices[:, None], final_recon_input_views, final_generated_missing_views)
   
      comparison = jnp.concatenate([
          images1[:8].reshape(-1, h, w, 1),
          images2[:8].reshape(-1, h, w, 1),
          top_half_processed[:8].reshape(-1, h, w, 1),
          bottom_half_processed[:8].reshape(-1, h, w, 1),
          images1[8:16].reshape(-1, h, w, 1),
          images2[8:16].reshape(-1, h, w, 1),
          top_half_processed[8:16].reshape(-1, h, w, 1),
          bottom_half_processed[8:16].reshape(-1, h, w, 1),
          images1[16:24].reshape(-1, h, w, 1),
          images2[16:24].reshape(-1, h, w, 1),
          top_half_processed[16:24].reshape(-1, h, w, 1),
          bottom_half_processed[16:24].reshape(-1, h, w, 1),
          images1[24:32].reshape(-1, h, w, 1),
          images2[24:32].reshape(-1, h, w, 1),
          top_half_processed[24:32].reshape(-1, h, w, 1),
          bottom_half_processed[24:32].reshape(-1, h, w, 1)
      ])
      comparison_masked = jnp.concatenate([
          input_x1_masked[:8].reshape(-1, h, w, 1),
          input_x2_masked[:8].reshape(-1, h, w, 1),
          top_half_withoriginal[:8].reshape(-1, h, w, 1),
          bottom_half_withoriginal[:8].reshape(-1, h, w, 1),
          input_x1_masked[8:16].reshape(-1, h, w, 1),
          input_x2_masked[8:16].reshape(-1, h, w, 1),
          top_half_withoriginal[8:16].reshape(-1, h, w, 1),
          bottom_half_withoriginal[8:16].reshape(-1, h, w, 1),
          input_x1_masked[16:24].reshape(-1, h, w, 1),
          input_x2_masked[16:24].reshape(-1, h, w, 1),
          top_half_withoriginal[16:24].reshape(-1, h, w, 1),
          bottom_half_withoriginal[16:24].reshape(-1, h, w, 1),
          input_x1_masked[24:32].reshape(-1, h, w, 1),
          input_x2_masked[24:32].reshape(-1, h, w, 1),
          top_half_withoriginal[24:32].reshape(-1, h, w, 1),
          bottom_half_withoriginal[24:32].reshape(-1, h, w, 1)
      ])
      
      generate_images1, generate_images2 = vae.generate(z_1, z_2)
      generate_images1 = generate_images1.reshape(-1, h, w, 1)
      generate_images2 = generate_images2.reshape(-1, h, w, 1)
    else:
      comparison = jnp.zeros((1,1))
      generate_images1 = jnp.zeros((1,1))
      generate_images2 = jnp.zeros((1,1))
    
    return metrics_iter, comparison, generate_images1, generate_images2, ll, \
           final_recon_input_views, final_generated_missing_views, \
           final_original_input_views, final_original_missing_views, final_mat, mask_choices, comparison_masked
  return nn.apply(eval_model, vae.model(latents, num_out, alpha))({'params': params})
eval_generation_f = jax.jit(eval_generation, static_argnums=(6,7, 8, 9, 10,11,12,13,14,15,16))


def get_z(params, images1, images2, z_rng, latents,num_out, alpha):
  def eval_model(vae):
    recon_images1, mean1, logvar1, z1, recon_images2, mean2, logvar2, z2, _= vae(images1, images2, z_rng)
    return z1, recon_images1, z2, recon_images2, mean1, mean2, logvar1, logvar2
  # zs1, recons1, zs2, recons2, mean1, = nn.apply(eval_model, vae.model(latents))({'params': params})
  zs1, recons1, zs2, recons2, mean1, mean2, logvar1, logvar2, _ = nn.apply(eval_model, vae.model(latents, num_out, alpha))({'params': params})
  # return {'z_v1':zs1, 'z_v2':zs2, 'rec_v1':recons1, 'rec_v2':recons2}
  return {'z_v1':zs1, 'z_v2':zs2, 'rec_v1':recons1, 'rec_v2':recons2, 'mean1':mean1, 'mean2':mean2, 'logvar1':logvar1, 'logvar2':logvar2}

def get_stats(params, images1, images2, z_rng, latents, num_out, alpha):
  def eval_model(vae):
    _, mean1, logvar1, _, _, mean2, logvar2, _, mat= vae(images1, images2, z_rng)
    return mean1, mean2, logvar1, logvar2, mat
  mean1, mean2, logvar1, logvar2, mat  = nn.apply(eval_model, vae.model(latents, num_out, alpha))({'params': params})
  return mean1, mean2, logvar1, logvar2, mat

def get_all_z(params, images1, images2, z_rng, latents, num_out, alpha, binary):
  def eval_model(vae):
    recon_images1, mean1, _, z1, recon_images2, mean2, _, z2, _= vae(images1, images2, z_rng)
    return z1, recon_images1, z2, recon_images2, mean1, mean2
    # recon_images1, mean1, _, _, recon_images2, mean2, _, _= vae(images1, images2, z_rng)
    # return mean1, recon_images1, mean2, recon_images2

  zs1, recons1, zs2, recons2, mean1, mean2 = nn.apply(eval_model, vae.model(latents, num_out, alpha))({'params': params})
  def apply_dec(vae):
    z1_cond, recon_x1, z2_cond, recon_x2 = vae.cross_gen(zs1, zs2, z_rng)
    return z1_cond, recon_x1, z2_cond, recon_x2

  z1_cond, recon_x1, z2_cond, recon_x2 = nn.apply(apply_dec, vae.model(latents, num_out, alpha))({'params': params})
  full_res = {'z_v1':zs1, 'z_v2':zs2, 'rec_v1':recons1, 'rec_v2':recons2, 'mean1':mean1, 'mean2':mean2}
  cond_res = {'z1_cond':z1_cond, 'recon_x1':recon_x1, 'z2_cond':z2_cond, 'recon_x2':recon_x2}

  #calculate losses
  if binary:
    loss1 = jnp.mean(met.bce_probs(recons1, images1))
    loss2 = jnp.mean(met.bce_probs(recons2, images2))
    loss_z1 = jnp.mean(met.bce_probs(recon_x1, images1))
    loss_z2 = jnp.mean(met.bce_probs(recon_x2, images2))
  else:
    loss1 = jnp.mean(met.mse_loss(recons1, images1))
    loss2 = jnp.mean(met.mse_loss(recons2, images2))
    loss_z1 = jnp.mean(met.mse_loss(recon_x1, images1))
    loss_z2 = jnp.mean(met.mse_loss(recon_x2, images2))
  # losses = {'loss1':loss1, 'loss2':loss2, 'loss_z1':loss_z1, 'loss_z2':loss_z2}
  losses = jnp.array([loss1, loss2, loss_z1, loss_z2])
  return full_res, cond_res, losses

# # Define a helper function to rotate a single 2D image with a given angle.
# # Define a helper function to rotate a single 2D image with a given angle.
def _rotate_single_image_with_angle(image_2d_array, angle):
   """
   Rotates a single 2D image by a given angle using scipy.ndimage.rotate,
   wrapped with jax.pure_callback to make it compatible with JAX transformations (like vmap).
   """
   # Define the shape and dtype of the output for jax.pure_callback
   output_shape = image_2d_array.shape
   output_dtype = image_2d_array.dtype

   # Python function that performs the rotation using scipy
   # This function will receive concrete NumPy arrays from jax.pure_callback
   def _rotate_python_func(image_np, angle_np):
       return scipy.ndimage.rotate(
           input=image_np,
           angle=angle_np,
           reshape=False,       # Keep the output shape the same as input (28x28)
           #mode='nearest'       # How to handle points outside the boundaries (e.g., fill with nearest value)
           
           mode='constant', # Use 'constant' mode (fill with black)
           cval=0.0,        # Fill value for 'constant' mode (black for 0-1 images)
           order=1          # Use linear interpolation (order=1) or cubic (order=3)
                            # order=0 is nearest-neighbor. Higher orders are smoother.
       )

   # This tells JAX to run the Python function with concrete values
   # when it encounters it during tracing, and then continue tracing
   # with the result. It essentially breaks the tracing through scipy.
   rotated_jax_array = jax.pure_callback(
       _rotate_python_func,
       jax.ShapeDtypeStruct(output_shape, output_dtype), # Specify output shape and dtype
       image_2d_array, # Input JAX array 
       angle           # Input JAX array (will be converted to NumPy inside callback)
   )
   return rotated_jax_array


def train_and_eval_split(config: ml_collections.ConfigDict, train_ds, val_ds, test_ds, full_train,full_val,
                         num_examples, key_val, zero = False, binary=True, orthog = True, param = True, 
                         cut = 0.02, print_images = False, toy = False, lambda_val =20, results_dir=None, figures_dir=None):
  """Train and evaulate pipeline."""
  rng = random.key(key_val)
  rng, key = random.split(rng)

  init_data = jnp.ones((config.batch_size, config.num_out[0]), jnp.float32)
  params = vae.model(config.latents, config.num_out, config.alpha).init(key, init_data, init_data, rng)['params']

  state = train_state.TrainState.create(
      apply_fn=vae.model(config.latents, config.num_out, config.alpha).apply,
      params=params,
      tx=optax.adam(config.learning_rate),
  )

  rng, z_key, eval_rng = random.split(rng, 3)
  steps_per_epoch = (
      num_examples // config.batch_size
    )

  if zero:
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero,lambda_va):
        return ts.train_step_noC(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero)
  elif (not binary) & orthog & param: #cont., orthog by param
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero,lambda_va):
        return ts.train_step(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero)
  elif (not binary) & orthog & (not param): 
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero, lambda_val):
        return ts.train_step_pen(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero, lambda_val)
  elif binary & orthog & param: #binary, orthog by param
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero, lambda_val):
        return ts.train_step_bin(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero)
  else: #binary, orthog with a penalty
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero, lambda_val):
        return ts.train_step_bin_pen(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero, lambda_val)
    
  
  beta_vec = other.make_beta(steps_per_epoch,1, 1)
  if toy:
    test_ds1 = jnp.expand_dims(jnp.array(test_ds[:,0]),axis=1)
    test_ds2 = jnp.expand_dims(jnp.array(test_ds[:,1]),axis=1)
    val_ds1 = jnp.expand_dims(jnp.array(val_ds[:,0]),axis=1)
    val_ds2 = jnp.expand_dims(jnp.array(val_ds[:,1]),axis=1)
    full_train1 = jnp.expand_dims(jnp.array(full_train[:, 0]), axis=1)
    full_train2 = jnp.expand_dims(jnp.array(full_train[:, 1]), axis=1)
    z_1 = 1
    z_2 = 1
  else:
    # Reshape the flattened test_ds and val_ds (num_test_images, 784) back to 2D (num_test_images, 28, 28).
    images_2d_test = jnp.reshape(test_ds, (test_ds.shape[0], 28, 28)).astype(jnp.float32)
    images_2d_val = jnp.reshape(val_ds, (val_ds.shape[0], 28, 28)).astype(jnp.float32)

    # Define the repeating rotation angles pattern.
    rotation_angles_pattern = jnp.array([0., 0., 0., 0.])

    # Create an array of specific angles for each image in the test and val set.
    test_image_indices = jnp.arange(images_2d_test.shape[0])
    angles_for_test_set = rotation_angles_pattern[test_image_indices % len(rotation_angles_pattern)]
    val_image_indices = jnp.arange(images_2d_val.shape[0])
    angles_for_val_set = rotation_angles_pattern[val_image_indices % len(rotation_angles_pattern)]

    # Apply the rotation to the entire test and val set.
    rotated_images_2d_test = vmap(_rotate_single_image_with_angle)(images_2d_test, angles_for_test_set)
    rotated_images_2d_val = vmap(_rotate_single_image_with_angle)(images_2d_val, angles_for_val_set)

    # Flatten the rotated 2D test images back to 1D (num_test_images, 784).
    flattened_rotated_test_set = jnp.reshape(rotated_images_2d_test, (test_ds.shape[0], -1))
    flattened_rotated_val_set = jnp.reshape(rotated_images_2d_val, (val_ds.shape[0], -1))
    
    test_ds_bin = other.binarize_array(flattened_rotated_test_set, cut)
    test_ds1 = test_ds_bin[:,0:392]
    test_ds2 = test_ds_bin[:, 392:784]
    
    val_ds_bin = other.binarize_array(flattened_rotated_val_set, cut)
    val_ds1 = val_ds_bin[:,0:392]
    val_ds2 = val_ds_bin[:, 392:784]
    
    # Same process for the full train and full val sets
    full_train_2d = jnp.reshape(full_train, (full_train.shape[0], 28, 28)).astype(jnp.float32)
    full_train_indices = jnp.arange(full_train_2d.shape[0])
    angles_for_full_train = rotation_angles_pattern[full_train_indices % len(rotation_angles_pattern)]
    rotated_full_train_2d = vmap(_rotate_single_image_with_angle)(full_train_2d, angles_for_full_train)
    flattened_rotated_full_train = jnp.reshape(rotated_full_train_2d, (full_train.shape[0], -1))

    full_train_bin = other.binarize_array(flattened_rotated_full_train, cut)
    
    full_train1 = full_train_bin[:,0:392]
    full_train2 = full_train_bin[:, 392:784]
    
    full_val_2d = jnp.reshape(full_val, (full_val.shape[0], 28, 28)).astype(jnp.float32)
    full_val_indices = jnp.arange(full_val_2d.shape[0])
    angles_for_full_val = rotation_angles_pattern[full_val_indices % len(rotation_angles_pattern)]
    rotated_full_val_2d = vmap(_rotate_single_image_with_angle)(full_val_2d, angles_for_full_val)
    flattened_rotated_full_val = jnp.reshape(rotated_full_val_2d, (full_val.shape[0], -1))

    full_val_bin = other.binarize_array(flattened_rotated_full_val, cut)
    
    full_val1 = full_val_bin[:,0:392]
    full_val2 = full_val_bin[:, 392:784]
    
    z_1 = random.normal(z_key, (64, config.latents[0]))
    z_2 = random.normal(z_key, (64, config.latents[1]))

  for epoch in range(config.num_epochs):
    for i in range(steps_per_epoch):
      batch = next(train_ds) # batch here has shape (batch_size, 784)
      if toy:
        batch1 = jnp.expand_dims(jnp.array(batch[:,0]),axis=1)
        batch2 = jnp.expand_dims(jnp.array(batch[:, 1]), axis=1)
      else:
        # reshape the flattened batch (batch, 784) back to 2D (batch_size,28,28)
        images_2d = jnp.reshape(batch, (config.batch_size, 28, 28)).astype(jnp.float32)
        
        # Define the repeating rotation angles pattern
        rotation_angles_pattern = jnp.array([0., 0., 0., 0.])
        
        # Create an array of specific angles for each image in the current batch.
        # This uses numbers to repeat the pattern (0, 45, 90, 135, 0, 45, ...).
        batch_indices_pattern = jnp.arange(config.batch_size)
        angles_for_batch = rotation_angles_pattern[batch_indices_pattern % len(rotation_angles_pattern)]
        
        # Apply the rotation to the entire batch using jax.vmap.
        rotated_images_2d = vmap(_rotate_single_image_with_angle)(images_2d, angles_for_batch)
        # Flatten the rotated 2D images back to 1D (batch_size, 784)
        # This is necessary because `other.binarize_array` expects a flattened input.
        flattened_rotated_batch = jnp.reshape(rotated_images_2d, (config.batch_size, -1))
        
        batch_bin = other.binarize_array(flattened_rotated_batch,cut)
        batch1 = batch_bin[:,0:392]
        batch2 = batch_bin[:, 392:784]
      rng, key = random.split(rng)
      beta = float(beta_vec[i])
      state = train_step(state, batch1, batch2, key, config.latents, config.num_out, config.alpha,
                                beta,zero,lambda_val)
      # return get_stats(state.params, full_train1, full_train2, eval_rng, config.latents, config.num_out, config.alpha)
      # return get_all_z(state.params, full_train1, full_train2, eval_rng, config.latents, config.num_out, config.alpha, binary)
    metrics_val, comparison_val, sample1_val, sample2_val, ll_val, _, _, mat_val= eval_f(
        state.params, test_ds1, test_ds2, z_1, z_2, eval_rng, config.latents, config.num_out, config.alpha,  binary, param, zero, print_images =  True,
        results_dir=results_dir, figures_dir=figures_dir
    )
    if(print_images):
      if (epoch + 1) % 30 == 0:
        vae_utils.save_image(comparison_val, os.path.join(figures_dir, f'reconstruction_val_{epoch}.png'), nrow=8)
        vae_utils.save_image(sample1_val, os.path.join(figures_dir, f'sample1_val_{epoch}.png'), nrow=8)
        vae_utils.save_image(sample2_val, os.path.join(figures_dir, f'sample2_val_{epoch}.png'), nrow=8)
    if (epoch + 1) % 5 == 0:
      print(
          'eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}, LL: {:.4f}'.format(
              epoch + 1, metrics_val['loss'], metrics_val['bce'], metrics_val['kld'], ll_val)
      )

  final_key = random.key(key_val + config.num_epochs + 2000)

  metrics_test, comparison_test, sample1_test, sample2_test, ll_test, \
  final_recon_input_views_test, final_generated_missing_views_test, \
  final_original_input_views_test, final_original_missing_views_test, mat_test, mask_choices, comparison_masked= eval_generation_f(
        state.params, test_ds1, test_ds2, z_1, z_2, eval_rng, config.latents, config.num_out, config.alpha,  binary, param, zero, print_images = True,
        results_dir=results_dir, figures_dir=figures_dir, mask_rng= final_key
    )
    
  top_view_for_classifier = jnp.where(mask_choices[:, None], final_generated_missing_views_test, final_original_input_views_test)
  bottom_view_for_classifier = jnp.where(mask_choices[:, None], final_original_input_views_test, final_generated_missing_views_test)

  test_data_for_classifier_unmasked_plus_generated = jnp.concatenate(
    [top_view_for_classifier, bottom_view_for_classifier],
    axis=1) # Concate
  
  print(
      'Final Test Eval (Conditional Gen): loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}, LL: {:.4f}'.format(
          float(jnp.mean(metrics_test['loss'])), float(jnp.mean(metrics_test['bce'])),
          float(jnp.mean(metrics_test['kld'])), float(jnp.mean(ll_test)))
  )
  if(print_images): # Save final test images
      vae_utils.save_image(comparison_test, os.path.join(figures_dir, f'reconstruction_test_final.png'), nrow=8)
      vae_utils.save_image(comparison_masked, os.path.join(figures_dir, f'reconstruction_test_final_masked.png'), nrow=8)
      vae_utils.save_image(sample1_test, os.path.join(figures_dir, f'sample1_test_final.png'), nrow=8)
      vae_utils.save_image(sample2_test, os.path.join(figures_dir, f'sample2_test_final.png'), nrow=8)
  
  rng = random.key(key_val)
  full_res, full_cond, full_loss = get_all_z(state.params, full_train1, full_train2, rng, config.latents, config.num_out, config.alpha, binary)
  val_res, val_cond, val_loss = get_all_z(state.params, full_val1, full_val2, rng, config.latents, config.num_out, config.alpha, binary)

  return full_res, state.params,  val_res, full_cond, val_cond, full_loss, val_loss, mat_val, comparison_val,\
  test_data_for_classifier_unmasked_plus_generated, full_train1, full_train2, final_recon_input_views_test, final_generated_missing_views_test, \
  test_ds1, test_ds2, top_view_for_classifier, bottom_view_for_classifier, metrics_val, ll_val, metrics_test,ll_test
