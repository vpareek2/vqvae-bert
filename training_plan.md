# Single Phase Approach
Below is a **step-by-step** guide to get started with a **single-phase** (end-to-end) training approach where both the **VQ-VAE** (encoder, codebook, decoder) **and** the **Transformer** (text → code indices) learn **simultaneously**. This means you do **one training loop** that updates everything at once, rather than pre-training the VQ-VAE separately.

---

## 1. Define the Unified Model

You want to build a **combined** model or training loop that includes:

1. **VQ-VAE**:
   - **Encoder**: Maps an image \(x\) to a continuous latent \(z_e\).
   - **Quantizer / Codebook**: Maps \(z_e\) to discrete indices (and quantized vectors) \(z_q\).
   - **Decoder**: Reconstructs the image from \(z_q\).

2. **Transformer**:
   - Takes the **text** input (e.g., 3–5 tokens).
   - Predicts the **same code indices** that the VQ-VAE encoder+quantizer produces.

3. **Loss Head**:
   - **VQ-VAE Loss** = Reconstruction + Commitment (like your normal VQ-VAE).
   - **Transformer Cross-Entropy Loss** to match predicted code indices vs. actual (ground-truth) code indices.

Putting it all together, you have something like:

```
CombinedModel:
    def forward(image, text):
        # 1) Encode image -> z_e
        z_e = vqvae.encoder(image)
        z_e = vqvae.pre_quantization_conv(z_e)

        # 2) Discretize -> z_q, code_indices
        codebook_loss, z_q, perplexity, _, code_indices = vqvae.vector_quantization(z_e)

        # 3) Transformer predicts code indices from text
        predicted_logits = text_transformer(text)
          # shape might be (batch_size, #latent_positions, n_embeddings)

        # (Optional) Feed code_indices or z_q to the decoder
        x_hat = vqvae.decoder(z_q)

        return (x_hat, code_indices, predicted_logits, codebook_loss, perplexity)
```

---

## 2. Single-Phase Training Loop

Below is a high-level loop:

1. **Data Loader**: Each batch contains `(image, text)` pairs.
2. **Forward Pass**:
   1. Pass `image` into the VQ-VAE encoder + quantizer.
      - Get `z_q` (quantized latent) and the **ground-truth** code indices used by the codebook.
   2. Pass `text` into the Transformer to get **predicted logits** for each code index.
   3. Reconstruct from `z_q` via the VQ-VAE decoder → output `x_hat`.
3. **Compute Losses**:
   1. **VQ-VAE Loss**:
      \[
      \mathcal{L}_\text{vqvae} \;=\; \underbrace{\|x - x\_hat\|^2}_{\text{recon}} \;+\; \beta \cdot \underbrace{\|z\_q - \text{sg}[z\_e]\|^2}_{\text{commitment}}
      \]
   2. **Transformer Cross-Entropy**:
      \[
      \mathcal{L}_\text{xent} = \text{CrossEntropy}\bigl(\text{predicted\_logits}, \text{actual\_code\_indices}\bigr)
      \]
   3. **Combined Loss**:
      \[
      \mathcal{L}_\text{total} = \mathcal{L}_\text{vqvae} + \lambda \cdot \mathcal{L}_\text{xent}
      \]
      where \(\lambda\) is a scalar balancing how strongly you emphasize text→code alignment vs. pure reconstruction.

4. **Backward + Optimization**:
   1. `optimizer.zero_grad()`
   2. `loss_total.backward()`
   3. `optimizer.step()`

5. **Repeat** for multiple epochs over the dataset.

---

### 2.1 Handling Spatial Code Indices

If your latent space is `(batch, embedding_dim, height, width)`, then for each location \((h, w)\) you have a code index \(\in \{0,...,n\_embeddings-1\}\). That means the **Transformer** must predict a sequence of code indices, one for each \((h, w)\) in the latent grid.

1. **Flatten** the latent grid (e.g., `height * width = N_positions`).
2. **Transformer Output** can produce `N_positions` logits, each with dimension `n_embeddings`.
3. **Cross-Entropy** is done over `N_positions * batch_size` predictions.

**Implementation Detail**: You can flatten the code indices `[batch, height, width]` into `[batch, height*width]` and flatten your predicted logits accordingly.

---

## 3. Initialize & Watch for Instabilities

When starting from scratch, **both** the codebook and the text classifier are unknown. Early in training, the code indices might be chaotic. Two ways to mitigate potential instability:

1. **Reduce the Cross-Entropy Weight Initially**: Start with \(\lambda = 0.0\) or a very small value for some warm-up epochs. Let the VQ-VAE learn a stable codebook for, say, a few thousand steps. Then ramp up \(\lambda\).

2. **Lower Learning Rate** for Codebook Parameters: If your codebook embeddings move too fast, the Transformer cross-entropy predictions become a moving target, which can hamper convergence. Consider a smaller LR for the codebook or a separate optimizer group with a lower LR.

3. **Use “Teacher Forcing”** for the reconstruction:
   - **Ground-Truth Indices**: In the reconstruction path, feed the actual code indices from the encoder to the decoder. This ensures stable gradients for reconstruction.
   - Meanwhile, the **Transformer** is still predicting those same indices in parallel and is penalized if it’s wrong.
   - *Later*, you can experiment with decoding from the predicted indices if you want truly end-to-end training without teacher forcing.

---

## 4. Key Hyperparameters & Tuning

1. **\(\lambda\)** (Transformer Loss Weight):
   - Typically in the range \([0.1, 1.0]\).
   - If you see codebook usage collapses or recon quality drops, lower \(\lambda\). If the Transformer never learns to predict, raise \(\lambda\).

2. **\(\beta\)** (Commitment Cost):
   - If codebook usage is too low (only a few codes used), you might slightly increase \(\beta\). If usage is too high or codebook is chaotic, reduce it.

3. **Transformer Size**:
   - For short text (3–5 tokens), consider fewer layers/heads. E.g., 2–3 layers, 4 heads, embedding dim 128–256. This will speed up training and reduce overfitting.

4. **Optimizer Schedules**:
   - You can use a single optimizer for everything or separate parameter groups:
     - Group 1: VQ-VAE
     - Group 2: Transformer
     - Group 3: Codebook embeddings
   - Potentially apply different LR or weight decay.

---

## 5. Practical Example (Pseudo-Code)

Below is a rough pseudo-code snippet to illustrate a single-phase approach:

```python
model = CombinedModel(...)  # includes VQVAE + Transformer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

lambda_xent = 0.5  # example weighting
for epoch in range(num_epochs):
    for images, texts in dataloader:
        # Forward
        x_hat, code_indices, pred_logits, codebook_loss, perplexity = model(images, texts)

        # VQ-VAE recon loss
        recon_loss = F.mse_loss(x_hat, images)
        vqvae_loss = recon_loss + codebook_loss

        # XENT for text->indices
        # shape of code_indices: (batch_size, num_positions)
        # shape of pred_logits: (batch_size, num_positions, n_embeddings)
        xent_loss = F.cross_entropy(pred_logits.view(-1, n_embeddings),
                                    code_indices.view(-1))

        # Combined
        loss = vqvae_loss + lambda_xent * xent_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Log/print stats each epoch
    print(f"Epoch {epoch}: recon={recon_loss.item():.4f}, xent={xent_loss.item():.4f}, perplexity={perplexity:.2f}")
```

---

## 6. Monitoring & Debugging

- **Reconstruction Quality**: If `recon_loss` or validation PSNR/SSIM is poor or gets worse over time, you may need to reduce `lambda_xent`.
- **Transformer Accuracy**: Keep track of average cross-entropy or accuracy on code indices.
- **Codebook Usage**: If it collapses to a tiny subset of embeddings, try a smaller learning rate or freeze the codebook for some steps.
- **Visualize**:
  1. The reconstructed images `x_hat`.
  2. The code index maps.
  3. The text condition—are you collecting the correct text for each image?

---

### When Single-Phase Becomes Tricky

If you see the codebook drifting too wildly or the transformer loss never going down, you might implement a short warm-up that trains only the VQ-VAE (with `lambda_xent=0.0`) for the first few thousand steps. After some stability, enable the cross-entropy term. This still counts as “single-phase,” just with a short “ramp up” for the Transformer.

---

# Final Thoughts

That’s the **basic recipe** to get started with a single-phase, end-to-end approach. The key is to **keep everything stable**—both the codebook and the new text classifier—by careful weighting of losses, possible warm-ups, and monitoring codebook usage. If it works well, great! If not, consider switching to a **multi-phase** approach (pre-train VQ-VAE fully, then freeze the codebook when training the Transformer) for easier stability. Good luck!

# Multi-Stage Approach
Below is a **step-by-step** guide for the **multi-phase** (a.k.a. two- or three-phase) approach, which often gives more stable and predictable results than trying to train everything end-to-end at once. This approach is commonly used in “language-to-latent” tasks (text→VQ codes→decoder).

---

## 1. Overview of the Multi-Phase Approach

1. **Phase 1:** Train your **VQ-VAE** as if there were no Transformer at all.
   - Focus on getting good image reconstruction and stable codebook usage.
   - By the end of this phase, you have a well-trained VQ-VAE that can encode images into discrete code indices and decode them back.
2. **Phase 2:** **Freeze** the VQ-VAE’s encoder & codebook, and train the **Transformer** to predict the code indices for each image.
   - You can optionally keep the VQ-VAE decoder trainable, or freeze it as well—your call. Typically, you only need the decoder if you want to do a final check that the predicted indices reconstruct well.
3. **Phase 3 (Optional):** **Fine-tune** the entire pipeline end-to-end, unfreezing the VQ-VAE if you want them to co-adapt.
   - This phase can be short or skipped entirely. It sometimes yields slight improvements but can also reintroduce instability if not tuned carefully.

---

## 2. Phase 1: Train the VQ-VAE

### 2.1 VQ-VAE Training Loop
You likely already have this set up (as shown in your repo), but let’s restate the standard procedure:

1. **Initialize** your VQ-VAE (encoder, codebook, decoder).
2. For each batch of images:
   - Encode images → `z_e`
   - Vector Quantize `z_e` → `z_q`, codebook usage, perplexity, etc.
   - Decode `z_q` → `x_hat`
   - Compute:
     \[
       \mathcal{L}_{\text{VQ-VAE}} = \| x - x_{\hat} \|^2 + \beta \cdot \| z_q - \text{sg}[z_e] \|^2
     \]
   - Backprop and optimize.
3. Run multiple epochs until the reconstruction loss, codebook usage, perplexity, etc. converge to a stable range.

### 2.2 Typical Goals or “Done” Criteria
- **Reconstruction Loss**: Below a certain threshold you find acceptable (e.g., ~0.04–0.05 MSE for a normalized dataset, or whatever target you have).
- **Codebook Usage**: Not collapsing to <5% usage, hopefully ~10–30% usage or more, depending on your design.
- **Perplexity**: Stabilizes at a certain value (like ~50–200, depending on `n_embeddings`).

### 2.3 Save the Checkpoint
Once you’re satisfied with the VQ-VAE’s performance:
1. **Save** the entire VQ-VAE (encoder, codebook, decoder).
2. **Optionally** store the code embeddings in a separate checkpoint.
3. **Most Important**: You now have a stable codebook.

---

## 3. Phase 2: Freeze the VQ-VAE’s Encoder & Codebook, Train the Transformer

The goal now is: for each `(image, text)` pair, the Transformer must predict the code indices that the (already trained) VQ-VAE encoder+quantizer produce.

### 3.1 Preparing the Data

**Option A: Online Approach**
- For each batch, you still run the **frozen** VQ-VAE encoder+quantizer on the images to get the “ground-truth” code indices.
- Then feed the associated text to the Transformer to train a cross-entropy on those code indices.

**Option B: Offline Pre-Extraction** (Faster)
1. **Run** the entire training set through your **frozen** VQ-VAE encoder+quantizer.
2. **Save** the resulting code indices for each image.
3. Now you have a **(text, code_indices)** dataset, which is a standard classification problem for the Transformer:
   - For each (text, code_indices), predict code_indices from text.
4. This avoids re-running the VQ-VAE encoder each epoch, drastically speeding up training.

Either approach is fine, though the offline approach is usually simpler and more efficient if your dataset is not huge.

### 3.2 Transformer Training Loop

If you do **online**:
```python
model_vqvae.encoder.eval()
model_vqvae.vector_quantization.eval()
# Freeze parameters
for param in model_vqvae.encoder.parameters():
    param.requires_grad = False
for param in model_vqvae.vector_quantization.parameters():
    param.requires_grad = False

for batch in dataloader:
    images, text = batch

    # 1) Get code indices from the frozen VQ-VAE
    with torch.no_grad():
        z_e = model_vqvae.encoder(images)
        z_e = model_vqvae.pre_quantization_conv(z_e)
        _, z_q, _, _, code_indices = model_vqvae.vector_quantization(z_e)

    # 2) Transformer predicts code indices from text
    predicted_logits = transformer_model(text)
      # shape: (batch_size, num_positions, n_embeddings)

    # 3) Cross-entropy loss
    loss = xent_loss_fn(predicted_logits.view(-1, n_embeddings),
                        code_indices.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

If you do **offline**:
```python
# Step 1: Pre-Extract code_indices for all images
# Step 2: Build a (text, code_indices) dataset
# Step 3: Plain classification loop:
for text, code_indices in dataset:
    predicted_logits = transformer_model(text)
    loss = xent_loss_fn(predicted_logits.view(-1, n_embeddings),
                        code_indices.view(-1))
    # optimize ...
```

### 3.3 Monitoring

- **Transformer Cross-Entropy**: Should go down as it learns to predict the code indices.
- **Accuracy**: You can measure what fraction of the predicted code indices match the actual. If your latent grid is large, it might be more about top-k or partial correctness, but typically a simple average accuracy is a fine metric.
- **Time to Converge**: Usually faster than training a generative model from scratch because it’s “just classification.”

### 3.4 Optional Check: Reconstruction from Predicted Indices

Even if your VQ-VAE decoder is frozen, you can do an “inference check”:
1. Feed text to the Transformer.
2. Get predicted code indices.
3. Feed those indices (as z_q) into the VQ-VAE decoder to see the image reconstruction.
   - This helps verify the Transformer is predicting the correct codes.

If the images look good, your cross-entropy is likely quite decent.

---

## 4. Phase 3 (Optional): End-to-End Fine-Tuning

If you want to squeeze out extra synergy—maybe the codebook could rearrange itself to better serve text-based generation—you can unfreeze the entire pipeline and do a brief fine-tuning:

1. **Initialize** from your Phase 1 + Phase 2 checkpoints (the VQ-VAE weights and the Transformer weights).
2. **Unfreeze** some or all of the VQ-VAE (encoder, codebook, or decoder).
3. **Add** a combined loss:
   \[
     \mathcal{L}_\text{total} = \mathcal{L}_{\text{vqvae}} + \lambda \cdot \mathcal{L}_\text{xent}
   \]
4. **Train** for a small number of epochs or steps (e.g., 5–20% of the steps used in Phase 1).
   - Because the codebook is no longer fixed, it might rearrange to help the text mapping. But you must watch out for instability.

### 4.1 Why This Might Help
- If your text→latent alignment is good, but the codebook was learned without text in mind, letting the codebook shift slightly can produce latents more “natural” to the text model.
- This can yield improved generation fidelity or better code usage.

### 4.2 Why This Might Hurt
- If the codebook drastically reconfigures, the Transformer might get “confused” and degrade.
- This phase requires careful tuning of learning rates, loss weights, and short training schedules to avoid catastrophic forgetting.

---

## 5. Putting It All Together

### Phase 1: Train VQ-VAE Alone

```python
vqvae = VQVAE(...)
optimizer_vq = torch.optim.Adam(vqvae.parameters(), lr=...)
for epoch in range(...):
    for images in dataloader:
        recon_loss, x_hat, perplexity = vqvae(images)
        # standard VQ-VAE training
        ...
    # Evaluate recon, perplexity
# Save vqvae checkpoint
```

### Phase 2: Freeze VQ-VAE (Encoder & Codebook), Train Transformer

```python
# Load the trained VQ-VAE
vqvae.load_state_dict(...)
vqvae.encoder.requires_grad_(False)
vqvae.vector_quantization.requires_grad_(False)

transformer = TransformerModel(...)
optimizer_transformer = torch.optim.Adam(transformer.parameters(), lr=...)

for epoch in range(...):
    for (images, text) in dataloader:
        with torch.no_grad():
            z_e = vqvae.encoder(images)
            z_e = vqvae.pre_quantization_conv(z_e)
            _, z_q, _, _, code_indices = vqvae.vector_quantization(z_e)

        predicted_logits = transformer(text)
        xent_loss = cross_entropy(predicted_logits.view(-1, n_embeddings),
                                  code_indices.view(-1))

        optimizer_transformer.zero_grad()
        xent_loss.backward()
        optimizer_transformer.step()
    # Evaluate accuracy on code indices
# Save transformer checkpoint
```

### (Optional) Phase 3: Fine-Tune End-to-End

```python
# Reload both models
vqvae.load_state_dict(...)
transformer.load_state_dict(...)
combined_optimizer = torch.optim.Adam(
    list(vqvae.parameters()) + list(transformer.parameters()), lr=...
)

for epoch in range(...):
    for (images, text) in dataloader:
        # forward pass
        z_e = vqvae.encoder(images)
        z_e_conv = vqvae.pre_quantization_conv(z_e)
        codebook_loss, z_q, perplexity, _, code_indices = vqvae.vector_quantization(z_e_conv)
        x_hat = vqvae.decoder(z_q)

        pred_logits = transformer(text)
        xent_loss = cross_entropy(pred_logits.view(-1, n_embeddings),
                                  code_indices.view(-1))

        recon_loss = F.mse_loss(x_hat, images)
        vqvae_loss = recon_loss + codebook_loss
        total_loss = vqvae_loss + lambda_xent * xent_loss

        combined_optimizer.zero_grad()
        total_loss.backward()
        combined_optimizer.step()

    # Check recon, perplexity, code usage, transformer accuracy
```

---

## 6. Key Benefits of Multi-Phase

1. **Stable Codebook**: Phase 1 ensures your codebook is well-trained for image reconstruction.
2. **Simplified Classification**: Phase 2 is just a straightforward text→codes classification task (cross-entropy).
3. **Efficiency**: If you do offline extraction, Phase 2 training is quite fast.
4. **Cleaner Debugging**: Each phase has simpler goals and fewer moving parts.
5. **Easier Monitoring**: You can see if the codebook alone is good, then if the Transformer alone is good, before combining them.

**Main Downside**: You do not get a “fully end-to-end” system from scratch, so if you believe the codebook should be heavily influenced by text semantics, you might eventually want a small end-to-end fine-tune (Phase 3) after you’ve stabilized the simpler tasks.

---

# Final Summary

With the **multi-phase** approach:

1. **Phase 1**: **Train VQ-VAE** thoroughly → stable codebook usage, good recon.
2. **Phase 2**: **Freeze** the VQ-VAE’s encoder + codebook → Train the **Transformer** to classify each image’s code indices from text.
3. **(Optional) Phase 3**: **Unfreeze** and fine-tune the entire pipeline end-to-end, if desired, for final synergy.

This stepwise progression keeps training more stable and makes it much easier to debug and measure each part’s performance. Good luck!
