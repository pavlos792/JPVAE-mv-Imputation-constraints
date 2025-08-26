# classifier_AE_Clustering_pipeline_fixed5.py
"""
Clustering pipeline with a Flax/JAX autoencoder and KMeans.
- Fixes ConcretizationTypeError by using static image_side.
- Avoids passing Python objects into jitted fns by CLOSING OVER `model`.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import optax
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


# -----------------------------
# Model definition
# -----------------------------

class Encoder(nn.Module):
    latent_dim: int
    image_side: int            # e.g., 28 for 28x28
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool):
        # Expect x as (B, HW) or (B, H, W, C). If flat, reshape using static image_side.
        if x.ndim == 2:
            b = x.shape[0]
            x = x.reshape((b, self.image_side, self.image_side, 1))

        x = nn.Conv(32, (3, 3), strides=(2, 2), padding="SAME", name="enc_conv1")(x)  # 28->14
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train, name="enc_bn1")(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME", name="enc_conv2")(x)  # 14->7
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train, name="enc_bn2")(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = x.reshape((x.shape[0], -1))  # (B, 7*7*64)
        emb = nn.Dense(self.latent_dim, name="embedding_fc")(x)
        return emb


class Decoder(nn.Module):
    image_side: int            # used only to decide target flatten size; deconv architecture is fixed
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, z, train: bool):
        x = nn.Dense(7 * 7 * 64, name="dec_fc")(z)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], 7, 7, 64))

        x = nn.ConvTranspose(32, (3, 3), strides=(2, 2), padding="SAME", name="dec_deconv1")(x)  # 7->14
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train, name="dec_bn1")(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = nn.ConvTranspose(1, (3, 3), strides=(2, 2), padding="SAME", name="dec_deconv2")(x)   # 14->28
        x = nn.sigmoid(x)  # recon in [0,1]
        x = x.reshape((x.shape[0], self.image_side * self.image_side))  # (B, HW)
        return x


class Autoencoder(nn.Module):
    latent_dim: int
    image_side: int
    dropout_rate: float = 0.0

    def setup(self):
        self.encoder = Encoder(self.latent_dim, self.image_side, self.dropout_rate)
        self.decoder = Decoder(self.image_side, self.dropout_rate)

    def __call__(self, x, *, train: bool, rng: Optional[jax.Array] = None):
        emb = self.encoder(x, train=train)
        recon = self.decoder(emb, train=train)
        return emb, recon


# -----------------------------
# Loss, update, and training
# -----------------------------

def recon_loss_mse(recon_flat: jax.Array, target_flat: jax.Array) -> jax.Array:
    return jnp.mean(jnp.sum((recon_flat - target_flat) ** 2, axis=1))


def init_model(rng, input_shape: Tuple[int, ...], latent_dim: int, image_side: int, dropout_rate: float = 0.1):
    model = Autoencoder(latent_dim=latent_dim, image_side=image_side, dropout_rate=dropout_rate)
    dummy = jnp.zeros(input_shape, dtype=jnp.float32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, train=True, rng=rng)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})
    return model, params, batch_stats


def batch_generator(data: jax.Array, labels: Optional[jax.Array], batch_size: int, key: jax.Array):
    """Yield shuffled mini-batches with a JAX key."""
    n = data.shape[0]
    idx_all = jnp.arange(n)
    while True:
        key, sub = random.split(key)
        perm = random.permutation(sub, idx_all)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = data[idx]
            if labels is not None:
                yb = labels[idx]
                yield xb, yb
            else:
                yield xb, None


def train_autoencoder(
    data: jax.Array,
    num_epochs: int,
    batch_size: int,
    latent_dim: int,
    image_side: int,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.1,
    seed: int = 0,
):
    key = random.PRNGKey(seed)
    model, params, batch_stats = init_model(key, input_shape=(1, data.shape[1]), latent_dim=latent_dim, image_side=image_side, dropout_rate=dropout_rate)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Define the jitted step *inside* so it can close over `model` and `optimizer`.
    @jax.jit
    def autoencoder_step(params, batch_stats, opt_state, x_batch, rng):
        def loss_fn(p):
            variables = {"params": p, "batch_stats": batch_stats}
            (emb, recon), new_state = model.apply(
                variables, x_batch, train=True, rngs={"dropout": rng}, mutable=["batch_stats"]
            )
            loss = recon_loss_mse(recon, x_batch.reshape((x_batch.shape[0], -1)))
            return loss, (new_state, emb, recon)

        grads, (new_state, emb, recon) = jax.grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_loss, _ = loss_fn(new_params)
        return new_params, new_state["batch_stats"], new_opt_state, new_loss

    key, gen_key = random.split(key)
    gen = batch_generator(data, None, batch_size, gen_key)
    steps_per_epoch = max(1, data.shape[0] // batch_size)

    for epoch in range(num_epochs):
        key, ekey = random.split(key)
        total = 0.0
        for step in range(steps_per_epoch):
            step_key = random.fold_in(ekey, step)
            x_batch, _ = next(gen)
            params, batch_stats, opt_state, loss_val = autoencoder_step(
                params, batch_stats, opt_state, x_batch, step_key
            )
            total += float(loss_val)
        avg = total / steps_per_epoch
        print(f"[AE] Epoch {epoch+1}/{num_epochs} - recon MSE: {avg:.6f}")

    return model, params, batch_stats


# -----------------------------
# Embedding extraction
# -----------------------------

def make_forward_embed(model: Autoencoder):
    """Return a jitted function that closes over `model` (no Python objects as args)."""
    @jax.jit
    def _forward(params, batch_stats, x):
        variables = {"params": params, "batch_stats": batch_stats}
        emb, recon = model.apply(variables, x, train=False, rngs={})
        return emb, recon
    return _forward

def get_embeddings(params, batch_stats, model: Autoencoder, data: jax.Array, batch_size: int = 512):
    fwd = make_forward_embed(model)
    n = data.shape[0]
    out = []
    for i in range(0, n, batch_size):
        xb = data[i:i+batch_size]
        emb, _ = fwd(params, batch_stats, xb)
        out.append(np.array(emb))
    return np.concatenate(out, axis=0)


# -----------------------------
# Utilities
# -----------------------------

def _concat_halves(top, bottom):
    return jnp.concatenate([jnp.asarray(top, dtype=jnp.float32),
                            jnp.asarray(bottom, dtype=jnp.float32)], axis=1)

def _cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Hungarian-matched clustering accuracy."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[p, t] += 1
    cost = cm.max() - cm
    r_ind, c_ind = linear_sum_assignment(cost)
    matched = cm[r_ind, c_ind].sum()
    return float(matched) / float(len(y_true))


# -----------------------------
# Main API: clustering_results_combine
# -----------------------------

def clustering_results_combine(train_ds1, train_ds2, train_labs,
                               original_top_view_test, original_bottom_view_test,
                               processed_top_view_test, processed_bottom_view_test,
                               test_labs,
                               full_train1, full_train2, full_train_labs,
                               params=None,               # unused, kept for signature parity
                               num_epochs=10, batch_num=128, n_targets=10,
                               step_size=1e-3, start_range=0, end_range=784,
                               latent_dim=64, seed=0, dropout_rate=0.1) -> Dict[str, float]:
    """
    Train the autoencoder on full original training images, then run KMeans on embeddings and
    report Hungarian-matched clustering accuracies for each requested test set combination.
    Assumes inputs are in [0,1]; NO standardization is applied.
    """
    # 1) Build full images
    train_full = _concat_halves(train_ds1, train_ds2)[:, start_range:end_range]
    test_full_original = _concat_halves(original_top_view_test, original_bottom_view_test)[:, start_range:end_range]
    test_orig_top_gen_bottom = _concat_halves(original_top_view_test, processed_bottom_view_test)[:, start_range:end_range]
    test_gen_top_orig_bottom = _concat_halves(processed_top_view_test, original_bottom_view_test)[:, start_range:end_range]
    test_full_processed = _concat_halves(processed_top_view_test, processed_bottom_view_test)[:, start_range:end_range]

    # Ensure float32
    train_full = jnp.asarray(train_full, dtype=jnp.float32)
    test_full_original = jnp.asarray(test_full_original, dtype=jnp.float32)
    test_orig_top_gen_bottom = jnp.asarray(test_orig_top_gen_bottom, dtype=jnp.float32)
    test_gen_top_orig_bottom = jnp.asarray(test_gen_top_orig_bottom, dtype=jnp.float32)
    test_full_processed = jnp.asarray(test_full_processed, dtype=jnp.float32)

    # 2) Derive static image_side from feature length (must be a perfect square, e.g., 784 -> 28)
    feat_dim = int(train_full.shape[1])
    side_f = int(np.sqrt(feat_dim))
    if side_f * side_f != feat_dim:
        raise ValueError(f"Input feature length {feat_dim} is not a perfect square; got side {side_f}. "
                         "This conv AE expects square images; adjust start/end_range or swap to an MLP AE.")

    # 3) Train AE on train_full
    model, params_ae, batch_stats = train_autoencoder(
        data=train_full,
        num_epochs=num_epochs,
        batch_size=batch_num,
        latent_dim=latent_dim,
        image_side=side_f,
        learning_rate=step_size,
        dropout_rate=dropout_rate,
        seed=seed,
    )

    # 4) Embeddings
    train_emb = get_embeddings(params_ae, batch_stats, model, train_full, batch_size=512)
    test_emb_full = get_embeddings(params_ae, batch_stats, model, test_full_original, batch_size=512)
    test_emb_ot_gb = get_embeddings(params_ae, batch_stats, model, test_orig_top_gen_bottom, batch_size=512)
    test_emb_gt_ob = get_embeddings(params_ae, batch_stats, model, test_gen_top_orig_bottom, batch_size=512)
    test_emb_full_proc = get_embeddings(params_ae, batch_stats, model, test_full_processed, batch_size=512)

    # 5) KMeans on embeddings (no scaling)
    km = KMeans(n_clusters=int(n_targets), n_init=10, random_state=seed)
    train_pred = km.fit_predict(train_emb)
    pred_full  = km.predict(test_emb_full)
    pred_ot_gb = km.predict(test_emb_ot_gb)
    pred_gt_ob = km.predict(test_emb_gt_ob)
    pred_proc  = km.predict(test_emb_full_proc)

    # 6) Accuracies via Hungarian match
    train_labels = np.asarray(train_labs)
    test_labels  = np.asarray(test_labs)
    acc_train = _cluster_accuracy(train_labels, train_pred, int(n_targets))
    acc_full  = _cluster_accuracy(test_labels,  pred_full,  int(n_targets))
    acc_ot_gb = _cluster_accuracy(test_labels,  pred_ot_gb, int(n_targets))
    acc_gt_ob = _cluster_accuracy(test_labels,  pred_gt_ob, int(n_targets))
    acc_proc  = _cluster_accuracy(test_labels,  pred_proc, int(n_targets))

    metrics = {
        "train_full_original_acc": acc_train,
        "test_full_original_acc": acc_full,
        "test_orig_top_gen_bottom_acc": acc_ot_gb,
        "test_gen_top_orig_bottom_acc": acc_gt_ob,
        "test_full_processed_acc": acc_proc,
    }
    print("Clustering accuracies (Hungarian matched):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


# -----------------------------
# Example / sanity check (optional)
# -----------------------------

# if __name__ == "__main__":
    # # Dummy shapes to verify the pipeline runs; replace with your actual data.
    # key = random.PRNGKey(0)
    # Ntr, Nte = 512, 128
    # Hsplit = 392  # half of 28*28 = 784

    # # Fake halves in [0,1]
    # train_ds1 = random.uniform(key, (Ntr, Hsplit))
    # key, _ = random.split(key)
    # train_ds2 = random.uniform(key, (Ntr, Hsplit))
    # train_labs = np.random.randint(0, 10, size=(Ntr,))

    # original_top_view_test = random.uniform(key, (Nte, Hsplit))
    # key, _ = random.split(key)
    # original_bottom_view_test = random.uniform(key, (Nte, Hsplit))
    # processed_top_view_test = random.uniform(key, (Nte, Hsplit))
    # key, _ = random.split(key)
    # processed_bottom_view_test = random.uniform(key, (Nte, Hsplit))
    # test_labs = np.random.randint(0, 10, size=(Nte,))

    # # Unused in this clustering pipeline, but kept for signature parity
    # full_train1, full_train2, full_train_labs = train_ds1, train_ds2, train_labs

    # metrics = clustering_results_combine(
        # train_ds1, train_ds2, train_labs,
        # original_top_view_test, original_bottom_view_test,
        # processed_top_view_test, processed_bottom_view_test,
        # test_labs,
        # full_train1, full_train2, full_train_labs,
        # params=None,
        # num_epochs=2, batch_num=128, n_targets=10,
        # step_size=1e-3, start_range=0, end_range=784,
        # latent_dim=32, seed=0, dropout_rate=0.1
    # )
    # print(metrics)
