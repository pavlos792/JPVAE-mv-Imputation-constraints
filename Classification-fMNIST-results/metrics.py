import jax
import jax.numpy as jnp
from flax import linen as nn

# file containing metrics for elbo calculation 
# for both binary data - binary cross entropy (bce_with logits)
# and continuous data - mean squared error (mse_loss)

def _safe_clip_probs(p, eps=1e-6):
    # make sure we're working in floating point
    if p.dtype in (jnp.int32, jnp.int64):
        p = p.astype(jnp.float32)
    lo = jnp.asarray(eps, dtype=p.dtype)
    hi = jnp.nextafter(jnp.asarray(1.0, p.dtype), jnp.asarray(0.0, p.dtype))  # largest < 1
    return jnp.clip(p, lo, hi)

@jax.vmap
def bce_with_logits2(logits, labels):  
  logits1 = nn.log_sigmoid(logits)
  return -jnp.sum(labels * logits1 + (1. - labels) * jnp.log(-jnp.expm1(logits1)))

bce_with_logits = jax.jit(bce_with_logits2)

@jax.vmap
def bce_probs_batch(preds, labels, eps=1e-06):
  #p = jnp.clip(preds, eps, 1.0 - eps)
  p = _safe_clip_probs(preds, eps)
  labels = labels.astype(p.dtype)
  return -jnp.sum(labels * jnp.log(p) + (1. - labels) * jnp.log1p(-p))
  
bce_probs = jax.jit(bce_probs_batch)

@jax.vmap
def mse_loss2(original, reconstructed):
    return jnp.sum(jnp.square(original - reconstructed))

mse_loss = jax.jit(mse_loss2)

def kl(mean1, logvar1, mean2, logvar2, matrices):
  D1 = matrices['D1']
  D2 = matrices['D2']
  D1CT = matrices['D1CT']
  D2C = matrices['D2C']
  term1 = -jnp.sum(2 + logvar1) -jnp.sum(logvar2)
  means =  jnp.matmul(mean1,jnp.matmul(D1, mean1)) + jnp.matmul(mean2,jnp.matmul(D2, mean2))
  vars = jnp.trace(jnp.matmul(D1, jnp.diag(jnp.exp(logvar1)))) + jnp.trace(jnp.matmul(D2, jnp.diag(jnp.exp(logvar2))))
  term2 = -(matrices['log_detD'] + jnp.matmul(mean1,jnp.matmul(D1CT,mean2))+ jnp.matmul(mean2, jnp.matmul(D2C,mean1)))
  return 0.5*(means + term1 + vars + term2)

kl_div1 = jax.vmap(kl, in_axes=(0,0,0,0, None))
kl_div_og = jax.jit(kl_div1)


def kl_noC(mean1, logvar1, mean2, logvar2):
  term1 = -jnp.sum(2 + logvar1) -jnp.sum(logvar2)
  means =  jnp.matmul(mean1,mean1) + jnp.matmul(mean2,mean2)
  vars = jnp.sum(jnp.exp(logvar1)) + jnp.sum(jnp.exp(logvar2))
  term2 = -(jnp.log(mean1.shape[0]))
  return 0.5*(means + term1 + vars + term2)

kl_div_noC1 = jax.vmap(kl_noC, in_axes=(0,0,0,0))
kl_div_noC = jax.jit(kl_div_noC1)

def kl(mean1, logvar1, mean2, logvar2, C, no_latents):

  D1 = jnp.linalg.inv(jnp.eye(no_latents[0]) - jnp.matmul(jnp.transpose(C), C))
  D2 = jnp.linalg.inv(jnp.eye(no_latents[1]) - jnp.matmul(C, jnp.transpose(C)))
  D1CT = jnp.matmul(D1,jnp.transpose(C))
  D2C = jnp.matmul(D2, C)
  term1 = -jnp.sum(2 + logvar1) -jnp.sum(logvar2)
  means =  jnp.matmul(mean1,jnp.matmul(D1, mean1)) + jnp.matmul(mean2,jnp.matmul(D2, mean2))
  vars = jnp.trace(jnp.matmul(D1, jnp.diag(jnp.exp(logvar1)))) + jnp.trace(jnp.matmul(D2, jnp.diag(jnp.exp(logvar2))))
  term2 = -(jnp.log(jnp.linalg.det(D1)) + jnp.matmul(mean1,jnp.matmul(D1CT,mean2))+ jnp.matmul(mean2, jnp.matmul(D2C,mean1)))
  return 0.5*(means + term1 + vars + term2)

kl_div1 = jax.vmap(kl, in_axes=(0,0,0,0, None, None))
kl_div = jax.jit(kl_div1, static_argnums=(5))

def compute_metrics_bin(recon_x1, recon_x2, batch1, batch2, mean1, logvar1, mean2, logvar2, matrices, average=True):
  
  bce_loss1 = bce_with_logits(recon_x1, batch1)
  bce_loss2 = bce_with_logits(recon_x2, batch2)
      
  kld_loss = (kl_div_og(mean1, logvar1, mean2, logvar2, matrices))
  bce_loss = bce_loss1 + bce_loss2
  loss = bce_loss + kld_loss
  if average:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': jnp.mean(loss)}
  else:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': (loss)}


def compute_metrics(recon_x1, recon_x2, batch1, batch2, mean1, logvar1, mean2, logvar2, matrices, average=True):
  mse_loss1 = (mse_loss(recon_x1, batch1))
  mse_loss2 = (mse_loss(recon_x2, batch2))
  kld_loss = (kl_div_og(mean1, logvar1, mean2, logvar2, matrices))
  bce_loss = mse_loss1 + mse_loss2
  loss = bce_loss + kld_loss
  if average:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': jnp.mean(loss)}
  else:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': (loss)}
  
def compute_metrics_zero(recon_x1, recon_x2, batch1, batch2, mean1, logvar1, mean2, logvar2, average=True):
  mse_loss1 = (bce_probs(recon_x1, batch1))
  mse_loss2 = (bce_probs(recon_x2, batch2))
  kld_loss = kl_div_noC(mean1, logvar1, mean2, logvar2)
  bce_loss = mse_loss1 + mse_loss2
  loss = bce_loss + kld_loss
  if average:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': jnp.mean(loss)}
  else:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': (loss)}
 
# def debug_show_image(img, name="img"):
    # # img could be flattened; reshape if needed
    # if img.ndim == 1:
        # side = int(jnp.sqrt(img.shape[0]))
        # img = img.reshape(side, side)
    # jax.debug.print("{} min {m} max {M}\n{}", 
                    # name, m=jnp.min(img), M=jnp.max(img), img=img)


# def bce_probs_debug(preds, labels, eps=1e-7):
    # jax.debug.print("bce_probs_debug: preds shape {}, labels shape {}", preds.shape, labels.shape)
    # jax.debug.print("preds: min {m} max {M} any_nan {n} any_inf {i}",
                    # m=jnp.min(preds), M=jnp.max(preds),
                    # n=jnp.any(jnp.isnan(preds)), i=jnp.any(jnp.isinf(preds)))
    # jax.debug.print("labels: any_nan {n} any_inf {i}",
                    # n=jnp.any(jnp.isnan(labels)), i=jnp.any(jnp.isinf(labels)))

    # p = jnp.clip(preds, eps, 1.0 - eps)
    # jax.debug.print("after clip p: min {m} max {M}", m=jnp.min(p), M=jnp.max(p))
    
    # pred1 = jnp.log(p)

    # term1 = labels * pred1
    # term2 = (1. - labels) * jnp.log(jnp.expm1(pred1))
    
    # debug_show_image(preds[0], "Predicted image 0")
    # debug_show_image(labels[0], "Label image 0")

    # jax.debug.print("term1 any_nan {n1} min {m1} max {M1}",
                    # n1=jnp.any(jnp.isnan(term1)), m1=jnp.min(term1), M1=jnp.max(term1))
    # jax.debug.print("term2 any_nan {n2} min {m2} max {M2}",
                    # n2=jnp.any(jnp.isnan(term2)), m2=jnp.min(term2), M2=jnp.max(term2))

    # loss = -jnp.sum(term1 + term2)  # per-example
    # jax.debug.print("bce_probs_debug loss any_nan {n} min {m} max {M}",
                    # n=jnp.any(jnp.isnan(loss)), m=jnp.min(loss), M=jnp.max(loss))
    # return loss


 
def compute_metrics_bin_test(recon_x1, recon_x2, batch1, batch2, mean1, logvar1, mean2, logvar2, matrices, average=True):
  
  bce_loss1 = bce_probs(recon_x1, batch1)
  bce_loss2 = bce_probs(recon_x2, batch2)
  
#  _ = bce_probs_debug(recon_x1[:2], batch1[:2])  # DEBUG few
#  _ = bce_probs_debug(recon_x2[:2], batch2[:2])
      
  kld_loss = (kl_div_og(mean1, logvar1, mean2, logvar2, matrices))
  bce_loss = bce_loss1 + bce_loss2
  loss = bce_loss + kld_loss
  if average:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': jnp.mean(loss)}
  else:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': (loss)}