import mediapy
import torch
import jax.numpy as jnp
import numpy as np  

def extract_video_features(batch, fprop_dtype, flax_model, loaded_state):
    batch = mediapy.to_float01(batch)
    batch = torch.from_numpy(batch).unsqueeze(0).numpy()
    batch = jnp.asarray(batch, dtype=fprop_dtype or jnp.float32)

    embeddings, _ = flax_model.apply(loaded_state, batch, train=False)
    embeddings = embeddings.squeeze(0)  # [T, D]
    embeddings = np.asarray(embeddings, dtype=np.float32)

    return embeddings