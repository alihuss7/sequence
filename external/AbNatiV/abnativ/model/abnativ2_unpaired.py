# (c) 2023 Sormannilab and Aubin Ramon
#
# AbNatiV model, Pytorch version
#
# ============================================================================

import os
from typing import Tuple
import math
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from datetime import datetime
from einops import rearrange

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from .vq import VectorQuantize
from .utils import find_optimal_cnn1d_padding, find_out_padding_cnn1d_transpose

import torch
from torch import nn 
from torch.nn import functional as F

from .alignment.aho_consensus import (
    cdr1_aho_indices,
    cdr2_aho_indices,
    cdr3_aho_indices,
    fr_aho_indices,
)

import pytorch_lightning as pl
from einops.layers.torch import Rearrange

# Perform real-time analysis during each validation epoch.
# Save the following metrics and scores:
# 1. Validation scores in 'val_scores.csv'
# 2. PSSM scores in 'pssm_scores.csv'
# 3. Accuracy, Loss, reconstruction error, and perplexity in a pickle file
# Additionally, plot the following:
# - Receiver Operating Characteristic (ROC) curves
# - Area Under the Curve (AUC) scores
from .analysis_functions import calculate_pssm_aucs, get_abnativ_scores_per_batch, evaluate_pssm, evaluate_portion_score


class RotaryEncoding(nn.Module):
    """
    Rotary positional embeddings applied to queries and keys in a transformer.

    paper: RoFormer: Enhanced Transformer with Rotary Position Embedding
    link: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        """
        :param head_dim: The per head embedding dimension of the query or key vectors.
        :param theta: Scaling factor for the rotary frequencies.
        """
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                f"Rotary encoding requires an even embedding dimension, "
                f"but received head_dim={head_dim}. Please use an even number."
            )

        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, v_i: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary positional encoding to the input vector.
        A 2D case is: (x, y) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]] @ (x, y)

        :param v_i: Input tensor of shape :code:`(batch_size, num_heads, seq_len, head_dim)`, where head_dim should
            match `head_dim` in the constructor.
        :return: Tensor with rotary positional encoding applied, of the same shape as input.
        """
        _, _, seq_len, _ = v_i.shape

        t = torch.arange(seq_len, device=v_i.device, dtype=v_i.dtype).float()
        angles = torch.einsum("i,j->ij", t, self.freqs)
        emb_sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        emb_cos = torch.cos(angles).repeat_interleave(2, dim=-1)

        v_rotated = (v_i * emb_cos) + (self.rotate_half(v_i) * emb_sin)
        return v_rotated

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotates the pairwise elements of the input tensor by 90 degrees.
        (x1, x2) -> (-x2, x1)

        Example:
        x = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        x = RotaryEncoding.rotate_half(x)
        x = tensor([-1,  0, -3,  2, -5,  4, -7,  6, -9,  8])
        """
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")


class GatedSelfAttention(nn.Module):
    """
    Gated self-attention layer with a rotary position embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.1,
        rotary_theta: float = 10000.0,
    ):
        """
        :param embed_dim: The dimensionality of the input/output embeddings.
        :param num_heads: The number of attention heads to use. The embedding dimension :code:`embed_dim`
            must be divisible by this number.
        :param dropout_p: Dropout probability for the attention weights.
        :param rotary_theta: Theta parameter for the rotary positional encoding.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} should be divisible by the number of heads {num_heads}."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.linear_no_bias_g = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False), nn.Dropout(dropout_p)
        )

        self.sigmoid = nn.Sigmoid()

        head_dim = embed_dim // num_heads
        self.rotary_encoding = RotaryEncoding(head_dim=head_dim, theta=rotary_theta)

    def forward(self, s_i: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single tensor of embeddings.

        :param s_i: :code:`(batch_size, seq_len, embed_dim)` tensor of embeddings.
        :return: Updated tensor of embeddings, same shape as the input.
        """

        # Input projections
        s_i = self.layer_norm(s_i)

        q_i = rearrange(self.linear_q(s_i), "b i (h d) -> b h i d", h=self.num_heads)
        k_i = rearrange(self.linear_k(s_i), "b i (h d) -> b h i d", h=self.num_heads)
        v_i = rearrange(self.linear_v(s_i), "b i (h d) -> b h i d", h=self.num_heads)
        g_i = rearrange(
            self.sigmoid(self.linear_no_bias_g(s_i)),
            "b i (h d) -> b h i d",
            h=self.num_heads,
        )

        # Rotary positional encoding
        q_i = self.rotary_encoding(q_i)
        k_i = self.rotary_encoding(k_i)

        # Attention
        attn_output = F.scaled_dot_product_attention(q_i, k_i, v_i) * g_i
        attn_output = rearrange(attn_output, "b h i d -> b i (h d)")

        # Output projections
        s_i = self.out(attn_output)
        return s_i


class SelfAttentionBlock(nn.Module):
    """A single block that performs self-attention followed by a SwiGLU transition layer."""

    def __init__(
        self,
        d_embedding: int,
        num_heads: int,
        d_ff: int,
        dropout_p: float = 0.1,
    ):
        """
        :param embed_dim: Dimension of the input embeddings.
        :param num_heads: Number of attention heads.
        :param dff: Dimension of the hiden layer in the SIWLU 
        :param dropout_p: Dropout probability for the attention weights.
        """
        super().__init__()

        self.attention = GatedSelfAttention(
            embed_dim=d_embedding,
            num_heads=num_heads,
            dropout_p=dropout_p,
        )

        self.linear_w = nn.Linear(d_embedding, d_ff)
        self.linear_v = nn.Linear(d_embedding, d_ff)
        self.d_emb_linear = nn.Linear(d_ff, d_embedding)
        self.swish = nn.SiLU()

        self.layernorm1 = nn.LayerNorm(d_embedding, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_embedding, eps=1e-6)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single tensor of embeddings.

        :param x: :code:`(batch_size, seq_len, embed_dim)` tensor of embeddings.
        :return: Tensor of updated embeddings, same shape as the input.
        """
        # Attention
        attn_output = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.layernorm1(x)

        # SWIGLU
        linear_output = self.d_emb_linear(self.swish(self.linear_w(x))*self.linear_v(x))
        x = x + self.dropout(linear_output)
        x = self.layernorm2(x) # (batch_size, input_seq_len, d_embedding) + residual 

        return x

    

class Encoder(nn.Module):
  def __init__(self, d_embedding, kernel, stride, num_heads, num_mha_layers, d_ff,
              length_seq, alphabet_size, dropout=0):
    super(Encoder, self).__init__()

    # CNN1d embedding
    self.l_red, self.padding = find_optimal_cnn1d_padding(L_in=length_seq, K=kernel, S=stride)
    self.cnn_embedding =  nn.Sequential(Rearrange('b l r -> b r l'),
                nn.Conv1d(alphabet_size, d_embedding, kernel_size=kernel, stride=stride, padding=self.padding),
                Rearrange('b r l -> b l r'))

    # Positional encoding
    self.en_dropout = nn.Dropout(dropout)

    # MHA blocks
    self.en_MHA_blocks = nn.ModuleList([SelfAttentionBlock(d_embedding, num_heads, d_ff, dropout)
                       for _ in range(num_mha_layers)])

  def forward(self, x) -> torch.Tensor: 
    """
    Args:
      x: Tensor, shape [batch_size, input_seq_len, alphabet_size]
    """
    # CNN1d Embedding
    h = self.cnn_embedding(x) # (batch_size, l_red, d_embedding)
    h = self.en_dropout(h) 

    # MHA blocks
    for i, l in enumerate(self.en_MHA_blocks):
      h = self.en_MHA_blocks[i](h) # (batch_size, l_red, d_embedding)
    
    return h


class Decoder(nn.Module):
  def __init__(self, d_embedding, kernel, stride, num_heads, num_mha_layers, d_ff,
                  length_seq, alphabet_size, dropout=0):
    super(Decoder, self).__init__()

    # Positional encoding
    self.l_red, self.padding = find_optimal_cnn1d_padding(L_in=length_seq, K=kernel, S=stride)
    self.de_dropout = nn.Dropout(dropout)

    # MHA blocks
    self.de_MHA_blocks = nn.ModuleList([SelfAttentionBlock(d_embedding, num_heads, d_ff, dropout)
                       for _ in range(num_mha_layers)])

    # Dense reconstruction
    self.dense_to_alphabet = nn.Linear(d_embedding, alphabet_size)
    self.dense_reconstruction = nn.Linear(alphabet_size*self.l_red, length_seq*alphabet_size)

    # CNN1d reconstruction
    self.out_pad = find_out_padding_cnn1d_transpose(L_obj=length_seq, L_in=self.l_red, K=kernel, S=stride, P=self.padding)
    self.cnn_reconstruction =  nn.Sequential(Rearrange('b l r -> b r l'),
                nn.ConvTranspose1d(d_embedding, alphabet_size, kernel_size=kernel, stride=stride, 
                              padding=self.padding, output_padding=self.out_pad),
                Rearrange('b r l -> b l r'))
    
  
  def forward(self, q) -> torch.Tensor:
    """
    Args:
      q: Tensor, shape [batch_size, l_red, d_embedding]
    """
    z = self.de_dropout(q) 

    # MHA blocks
    for i, l in enumerate(self.de_MHA_blocks):
      z = self.de_MHA_blocks[i](z) # (batch_size, l_red, d_embedding)

    pre_proj_rep = z 

    # CNN reconstruction 
    z = self.cnn_reconstruction(z) # (batch_size, input_seq_len, alphabet_size)
    z_recon = F.softmax(z, dim=-1)

    return z_recon, pre_proj_rep, z



class AbNatiV_Model(pl.LightningModule):
  def __init__(self, hparams: dict):
    super(AbNatiV_Model, self).__init__()

    # MODEL
    self.run_name = hparams["run_name"]

    self.encoder = Encoder(hparams['d_embedding'], hparams['kernel'], hparams['stride'], hparams['num_heads'], 
                            hparams['num_mha_layers'], hparams['d_ff'], hparams['length_seq'], 
                            hparams['alphabet_size'], dropout=hparams['drop'])

    self.decoder = Decoder(hparams['d_embedding'], hparams['kernel'], hparams['stride'], hparams['num_heads'], 
                            hparams['num_mha_layers'], hparams['d_ff'], hparams['length_seq'], 
                            hparams['alphabet_size'], dropout=hparams['drop'])

    self.vqvae = VectorQuantize(
            dim=hparams['d_embedding'],
            codebook_size=hparams['num_embeddings'],
            codebook_dim=hparams['embedding_dim_code_book'],
            decay=hparams['decay'],
            kmeans_init=True,
            commitment_weight=hparams['commitment_cost']
            )

    # TRAINING
    self.learning_rate = hparams['learning_rate']

    # LOSS 
    if "loss" in hparams: self.loss = hparams['loss']
    else: self.loss = 'mse'
    if "gamma" in hparams: self.gamma = hparams['gamma']
    else: self.gamma = 1
    if "lambda" in hparams: self.lambda_ = hparams['lambda']
    else: self.lambda_ = 1

    # REAL-TIME ANALYSIS
    self.batch_size = hparams['batch_size']
    self.validation_step_outputs = []
    if 'fp_diverse_test' in hparams:
      self.fp_diverse_test = hparams['fp_diverse_test']
      self.fp_pssm = hparams['fp_pssm']
      if 'fp_abnativ1_human' in hparams:
         self.fp_abnativ1_human = hparams['fp_abnativ1_human']
      else: 
         self.fp_abnativ1_human = None
         
      self.fp_abnativ1_pssm = hparams['fp_abnativ1_pssm']
      self.fp_mouse = hparams['fp_mouse']
      self.fp_rhesus = hparams['fp_rhesus']
      
      if 'fp_al_uf' in hparams:
        self.fp_al_uf = hparams['fp_al_uf']
        self.fp_uf_raw = hparams['fp_uf_raw']
      else: 
        self.fp_al_uf = None
        self.fp_uf_raw = None

      if 'fp_diverse_test_vk' in hparams:
        self.fp_diverse_test_vk = hparams['fp_diverse_test_vk']
        self.fp_pssm_vk = hparams['fp_pssm_vk']
        self.fp_abnativ1_pssm_vk = hparams['fp_abnativ1_pssm_vk']
        self.fp_mouse_vk = hparams['fp_mouse_vk']
        self.fp_rhesus_vk = hparams['fp_rhesus_vk']
        self.fp_test_vk = hparams['fp_test_vk']
        self.fp_test= hparams['fp_test']
      
      else: 
        self.fp_diverse_test_vk = None
        self.fp_pssm_vk = None
        self.fp_abnativ1_pssm_vk = None
        self.fp_mouse_vk = None
        self.fp_rhesus_vk = None
        self.fp_test_vk = None
        self.fp_test= None

    else:
      self.fp_diverse_test = None
      self.fp_pssm = None
      self.fp_abnativ1_pssm = None
      self.fp_abnativ1_human = None
      self.fp_mouse = None
      self.fp_rhesus = None
      self.fp_al_uf = None
      self.fp_uf_raw = None

    if "conservation_index" in hparams:
      self.conservation_index = hparams["conservation_index"]
    else: 
      self.conservation_index = None

    if "caa_freq" in hparams:
      self.aa_freq = hparams["aa_freq"]
    else: 
      self.aa_freq = None

    self.save_hyperparameters()

  def on_train_start(self):
    if self.conservation_index:
      with open(self.conservation_index, "rb") as f:
          conservation_indices = pickle.load(f)
          self.conservation_indices = torch.tensor(1 - np.array(conservation_indices))

    if self.aa_freq:
      with open(self.aa_freq, "rb") as f:
          aa_freq = pickle.load(f)
          logging.info(f"Working with aa_freq order: {''.join(aa_freq.keys())}. Ensure this matches the dataloader's aa order.")
          self.aa_freq = torch.tensor(list(aa_freq.values()), dtype=torch.float32).T  # (input_seq_len, alphabet_size)
    

  def forward(self, data) -> dict:
    inputs = data[:][0][:][:]
    m_inputs = data[:][1][:][:]

    x = self.encoder(m_inputs)
    vq_outputs = self.vqvae(x)
    x_recon, x_pre_cnn_proj, z_logits = self.decoder(vq_outputs['quantize_projected_out'])

    # Loss computing 
    recon_error_pposi = self.calculate_recon_error_pposi(inputs, x_recon)
    recon_error_pbe = torch.mean(recon_error_pposi, dim=1)
    lambda_vq_output = self.lambda_ * vq_outputs['loss_vq_commit_pbe']
    loss_pbe = torch.add(recon_error_pbe, lambda_vq_output)

    mse_pposi = self.calculate_MSE_recon_error_pposi(inputs, x_recon)

    return {
        'inputs': inputs, # (batch_size, input_seq_len, alphabet_size)
        'x_recon': x_recon, # (batch_size, input_seq_len, alphabet_size)
        'mse_pposi': mse_pposi, # (batch_size, input_seq_len)
        'recon_error_pposi': recon_error_pposi, # (batch_size, input_seq_len)
        'recon_error_pbe': recon_error_pbe, # (batch_size)
        'loss_pbe': loss_pbe, # (batch_size)
        'x_encoder_pre_vq': x, # (batch_size, l_red, dim)
        'x_decoder_pre_proj': x_pre_cnn_proj, # (batch_size, l_red, dim)
        'z_logits': z_logits, # (batch_size, input_seq_len, alphabet_size)
        **vq_outputs
    }

  def calculate_MSE_recon_error_pposi(self, inputs, x_recon, conservation_index=False, focal=False, alpha=False):
    recon_error_pres_pposi = F.mse_loss(x_recon, inputs, reduction='none')
    recon_error_pposi = torch.mean(recon_error_pres_pposi, dim=-1)
    
    if conservation_index:
        self.conservation_indices = self.conservation_indices.to(recon_error_pposi.device)
        conservation_term = self.conservation_indices ** self.gamma
        recon_error_pposi *= conservation_term
    
    if focal: 
        p_true = torch.gather(x_recon, -1, inputs.argmax(dim=-1, keepdim=True)).squeeze(-1)
        focal_term = (1 - p_true) ** self.gamma
        recon_error_pposi *= focal_term

    if alpha:
        self.aa_freq = self.aa_freq.to(recon_error_pposi.device)
        inputs = inputs.to(recon_error_pposi.device)
        index_true = inputs.argmax(dim=-1)
        true_aa_freq = self.aa_freq[torch.arange(recon_error_pposi.shape[-1]), index_true]
        alpha_term = (1 - true_aa_freq)
        recon_error_pposi *= alpha_term
    
    return recon_error_pposi

  def calculate_CE_recon_error_pposi(self, inputs, x_recon, conservation_index=False, focal=False):
    x_recon = x_recon.reshape(x_recon.shape[0], x_recon.shape[2], x_recon.shape[1])
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[2], inputs.shape[1])
    recon_error_pposi = F.cross_entropy(x_recon, inputs, reduction='none')

    if conservation_index:
        self.conservation_indices = self.conservation_indices.to(recon_error_pposi.device)
        conservation_term = self.conservation_indices ** self.gamma
        recon_error_pposi *= conservation_term
    
    if focal: 
        p_true = torch.gather(x_recon, -1, inputs.argmax(dim=-1, keepdim=True)).squeeze(-1)
        focal_term = (1 - p_true) ** self.gamma
        recon_error_pposi *= focal_term

    return recon_error_pposi

  def calculate_recon_error_pposi(self, inputs, x_recon):
    # MSE
    if self.loss == 'mse':
        recon_error_pposi = self.calculate_MSE_recon_error_pposi(inputs, x_recon)
    elif self.loss == 'conservation_mse':
        recon_error_pposi = self.calculate_MSE_recon_error_pposi(inputs, x_recon, conservation_index=True)
    elif self.loss == 'focal_mse':
        recon_error_pposi = self.calculate_MSE_recon_error_pposi(inputs, x_recon, focal=True)
    elif self.loss == 'alpha_focal_mse':
        recon_error_pposi = self.calculate_MSE_recon_error_pposi(inputs, x_recon, focal=True, alpha=True)

    # CE
    elif self.loss == 'ce':
        recon_error_pposi = self.calculate_CE_recon_error_pposi(inputs, x_recon)
    elif self.loss == 'conservation_ce':
        recon_error_pposi = self.calculate_CE_recon_error_pposi(inputs, x_recon, conservation_index=True)
    elif self.loss == 'focal_ce':
        recon_error_pposi = self.calculate_CE_recon_error_pposi(inputs, x_recon, focal=True)

    else:
        raise ValueError(f"Unsupported loss type: {self.loss}")
    
    return recon_error_pposi

  def configure_optimizers(self):
    optim_groups = list(self.encoder.parameters()) + \
                    list(self.decoder.parameters()) + \
                    list(self.vqvae.parameters()) 

    return torch.optim.AdamW(optim_groups, lr=self.learning_rate)

  def training_step(self, batch, batch_idx) -> float:
    if batch_idx % 1000 == 0:
      current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
      logging.info(f"Training on batch {batch_idx} at {current_time}")
    
    vqvae_output = self(batch)

    loss_vqvae = torch.mean(vqvae_output['loss_pbe'])
    self.log("train_loss_vqvae", loss_vqvae, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    loss_vq_commit = torch.mean(vqvae_output['loss_vq_commit_pbe'])
    self.log("train_loss_vq_commit", loss_vq_commit, on_step=True,on_epoch=True,prog_bar=True, logger=True)

    nmse_accuracy = torch.mean(vqvae_output['recon_error_pbe'])
    self.log("train_loss_nmse_recons", nmse_accuracy, on_step=True, on_epoch=True,prog_bar=True, logger=True)

    perplexity = vqvae_output['perplexity']
    self.log("train_perplexity", perplexity, on_step=True, on_epoch=True,prog_bar=True, logger=True)

    return loss_vqvae

  def on_train_epoch_start(self) -> None:
    current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    logging.info(f"Training Epoch {self.current_epoch} started at {current_time}")
    return

  def on_train_epoch_end(self) -> None:
    current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    logging.info(f"Training Epoch {self.current_epoch} ended at {current_time}")
    return 

  def validation_step(self, batch, batch_idx) -> None:
    """
    Perform a single validation step for a given batch during the validation epoch.

    This method evaluates a batch of sequences using the AbNatiV model, computes various nativeness scores,
    and appends the results to the validation step outputs.

    Notes
    -----
    The method performs the following steps:
    1. Initializes a dictionary to store the mean scores for the batch.
    2. Calls `get_abnativ_scores_per_batch` to evaluate the batch and compute nativeness scores.
    3. Constructs an output dictionary containing the current epoch, batch index, model mean outputs, and VHH scores.
    4. Appends the output dictionary to `self.validation_step_outputs`.
    """
    
    if batch_idx % 1000 == 0:
      current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
      logging.info(f"Validation on batch {batch_idx} at {current_time}")

    scored_data_dict_mean = defaultdict(list)
    # Get abnativ scores for this batch
    model_output = get_abnativ_scores_per_batch(
      model=self, 
      model_type='None', 
      batch=batch,
      scored_data_dict_mean=scored_data_dict_mean
    )

    mean_loss = torch.mean(model_output['loss_pbe'])
    mean_recon_error = torch.mean(model_output["recon_error_pbe"])

    inputs_seq = torch.argmax(model_output["inputs"], dim=-1)
    x_recon_seq = torch.argmax(model_output["x_recon"], dim=-1)
    mean_accuracy = torch.mean((inputs_seq == x_recon_seq).float()).item()

    mean_output = {
      "mean_loss": mean_loss,
      "mean_recon_error": mean_recon_error,
      "mean_accuracy": mean_accuracy,
      "perplexity": model_output["perplexity"]
    }

    score_to_keep = "AbNatiV None Score"
    scored_data_dict_mean = defaultdict(list, {score_to_keep: scored_data_dict_mean[score_to_keep]})

    output = {
      "epoch": self.current_epoch,
      "batch_num": batch_idx,
      "model_output": mean_output,
      "scores": scored_data_dict_mean,
    }
    self.validation_step_outputs.append(output)
    return 

  def log_and_save_validation_metrics(self, output_dir: str) -> None:
    # Extract outputs for the current epoch. Although validation_step_outputs should only contain scores for the current epoch, 
    # we filter them to ensure consistency and robustness.
    model_outputs = [
      out["model_output"]
      for out in self.validation_step_outputs
      if out["epoch"] == self.current_epoch
    ]

    val_losses = torch.Tensor([out["mean_loss"] for out in model_outputs])
    total_val_loss = torch.mean(val_losses)
    self.log('val_loss', total_val_loss, on_epoch=True, logger=True)

    val_accuracies = torch.Tensor([out['mean_recon_error'] for out in model_outputs])
    total_val_accuracy = torch.mean(val_accuracies)
    self.log('val_nmse_accuracy', total_val_accuracy, on_epoch=True, logger=True)

    val_perplexities = torch.Tensor([out['perplexity'] for out in model_outputs])
    total_val_perplexity = torch.mean(val_perplexities)
    self.log('val_perplexity', total_val_perplexity, on_epoch=True, logger=True)

    accuracies = torch.Tensor([out['mean_accuracy'] for out in model_outputs])
    total_val_accuracy = torch.mean(accuracies)
    self.log('val_accuracy', total_val_accuracy, on_epoch=True, logger=True)

    # Extract the specified keys from each dictionary in model_outputs
    with open(f'{output_dir}/val_model_outputs.pkl', 'wb') as f:
        pickle.dump(model_outputs, f)

  def save_scores_and_process_pssm(self, output_dir: str) -> None:
    """
    Analyze and save validation scores, score PSSM using the current model, and generate ROC and PR plots.

    This function performs the following actions:
    1. Analyzes and saves validation scores stored in self.validation_step_outputs
    2. Scores PSSM (Position-Specific Scoring Matrix) using the current model and saves the results.
    3. Runs ROC (Receiver Operating Characteristic) and PR (Precision-Recall) analyses between validation scores and PSSM scores, and saves the resulting plots.

    Parameters
    ----------
    output_dir : str
        The path to the output directory where results and plots will be saved.
    """

    # Extract scores for the current epoch. Although validation_step_outputs should only contain scores for the current epoch, 
    # we filter them to ensure consistency and robustness.
    epoch_scores = [
        out["scores"]
        for out in self.validation_step_outputs
        if out["epoch"] == self.current_epoch
    ]

    # Combine scores from all batches into a single dictionary for the current epoch.
    scored_data_dict_mean = defaultdict(list)
    for scores in epoch_scores:
        for key in scores:
            scored_data_dict_mean[key].extend(scores[key])

    # Create a pandas DataFrame for the epoch scores and save it as 'val_scores.csv' in the specified directory.
    df_mean_epoch = pd.DataFrame.from_dict(scored_data_dict_mean)
    df_mean_epoch.to_csv(f"{output_dir}/val_scores.csv")

    ### VALIDATION vs others ###
    if not self.fp_abnativ1_pssm_vk:
      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_epoch, 
        fp_pssm=self.fp_pssm, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR Val-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_epoch, 
        fp_pssm=self.fp_abnativ1_pssm, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR Val-AbNatiV1-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      if self.fp_abnativ1_human:
        pr_auc, roc_auc = calculate_pssm_aucs(
          model=self, 
          model_type='None', 
          df_mean_val=df_mean_epoch, 
          fp_pssm=self.fp_abnativ1_human, 
          batch_size=self.batch_size, 
          output_dir=output_dir
        )

        self.log('PR Val-AbNatiV1-Human', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_epoch, 
        fp_pssm=self.fp_mouse, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR Val-Mouse', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_epoch, 
        fp_pssm=self.fp_rhesus, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR Val-Rhesus', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_epoch, 
        fp_pssm=self.fp_diverse_test, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR Val-Diverse_Test', torch.tensor(pr_auc), on_epoch=True, logger=True)

      ### ABNATIV1 TEST vs others ###
      df_mean_diverse_test = evaluate_pssm(self, 'None', self.fp_diverse_test, self.batch_size)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test, 
        fp_pssm=self.fp_pssm, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR Diverse_Test-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test, 
        fp_pssm=self.fp_abnativ1_pssm, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR Diverse_Test-AbNatiV1-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      if self.fp_abnativ1_human:
        pr_auc, roc_auc = calculate_pssm_aucs(
          model=self, 
          model_type='None', 
          df_mean_val=df_mean_diverse_test, 
          fp_pssm=self.fp_abnativ1_human, 
          batch_size=self.batch_size, 
          output_dir=output_dir
        )

        self.log('PR Diverse_Test-AbNatiV1-Human', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test, 
        fp_pssm=self.fp_mouse, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR Diverse_Test-Mouse', torch.tensor(pr_auc), on_epoch=True, logger=True)
    
      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test, 
        fp_pssm=self.fp_rhesus, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR Diverse_Test-Rhesus', torch.tensor(pr_auc), on_epoch=True, logger=True)


    if self.fp_abnativ1_pssm_vk:
      #### VLambda ####
      df_mean_test = evaluate_pssm(self, 'None', self.fp_test, self.batch_size)

      pr_auc, roc_auc = calculate_pssm_aucs(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_test, 
      fp_pssm=self.fp_pssm, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

      self.log('PR LAMBDA Test-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_test, 
        fp_pssm=self.fp_abnativ1_pssm, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR LAMBDA Test-AbNatiV1-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      if self.fp_abnativ1_human:
        pr_auc, roc_auc = calculate_pssm_aucs(
          model=self, 
          model_type='None', 
          df_mean_val=df_mean_test, 
          fp_pssm=self.fp_abnativ1_human, 
          batch_size=self.batch_size, 
          output_dir=output_dir
        )

        self.log('PR LAMBDA Test-AbNatiV1-Human', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_test, 
        fp_pssm=self.fp_mouse, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR LAMBDA Test-Mouse', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_test, 
        fp_pssm=self.fp_rhesus, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR LAMBDA Test-Rhesus', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_test, 
        fp_pssm=self.fp_diverse_test, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR LAMBDA Test-Diverse_Test', torch.tensor(pr_auc), on_epoch=True, logger=True)

      ### ABNATIV1 TEST vs others ###
      df_mean_diverse_test = evaluate_pssm(self, 'None', self.fp_diverse_test, self.batch_size)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test, 
        fp_pssm=self.fp_pssm, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR LAMBDA Diverse_Test-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test, 
        fp_pssm=self.fp_abnativ1_pssm, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR LAMBDA Diverse_Test-AbNatiV1-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test, 
        fp_pssm=self.fp_mouse, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR LAMBDA Diverse_Test-Mouse', torch.tensor(pr_auc), on_epoch=True, logger=True)
    
      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test, 
        fp_pssm=self.fp_rhesus, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR LAMBDA Diverse_Test-Rhesus', torch.tensor(pr_auc), on_epoch=True, logger=True)

      #### VKappa ####
      df_mean_test_vk = evaluate_pssm(self, 'None', self.fp_test_vk, self.batch_size)

      pr_auc, roc_auc = calculate_pssm_aucs(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_test_vk, 
      fp_pssm=self.fp_pssm_vk, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

      self.log('PR KAPPA Test-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_test_vk, 
        fp_pssm=self.fp_abnativ1_pssm_vk, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR KAPPA Test-AbNatiV1-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      if self.fp_abnativ1_human:
        pr_auc, roc_auc = calculate_pssm_aucs(
          model=self, 
          model_type='None', 
          df_mean_val=df_mean_test_vk, 
          fp_pssm=self.fp_abnativ1_human_vk, 
          batch_size=self.batch_size, 
          output_dir=output_dir
        )

        self.log('PR KAPPA Test-AbNatiV1-Human', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_test_vk, 
        fp_pssm=self.fp_mouse_vk, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR KAPPA Test-Mouse', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_test_vk, 
        fp_pssm=self.fp_rhesus_vk, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR KAPPA Test-Rhesus', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_test_vk, 
        fp_pssm=self.fp_diverse_test_vk, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR KAPPA Test-Diverse_Test', torch.tensor(pr_auc), on_epoch=True, logger=True)

      ### ABNATIV1 TEST vs others ###
      df_mean_diverse_test_vk = evaluate_pssm(self, 'None', self.fp_diverse_test_vk, self.batch_size)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test_vk, 
        fp_pssm=self.fp_pssm_vk, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR KAPPA Diverse_Test-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test_vk, 
        fp_pssm=self.fp_abnativ1_pssm_vk, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR KAPPA Diverse_Test-AbNatiV1-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test_vk, 
        fp_pssm=self.fp_mouse_vk, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR KAPPA Diverse_Test-Mouse', torch.tensor(pr_auc), on_epoch=True, logger=True)
    
      pr_auc, roc_auc = calculate_pssm_aucs(
        model=self, 
        model_type='None', 
        df_mean_val=df_mean_diverse_test_vk, 
        fp_pssm=self.fp_rhesus_vk, 
        batch_size=self.batch_size, 
        output_dir=output_dir
      )

      self.log('PR KAPPA Diverse_Test-Rhesus', torch.tensor(pr_auc), on_epoch=True, logger=True)




  def run_uf_grafting(self, output_dir: str) -> None:
    """
    Run the UF grafting discrimination test
    Parameters
    ----------
    output_dir : str
        The path to the output directory where results and plots will be saved.
    """

    aho_selected_portion = list(cdr1_aho_indices) + list(cdr2_aho_indices) + list(cdr3_aho_indices)
    df_score = evaluate_portion_score(self, 'None', self.fp_al_uf, self.batch_size, aho_selected_portion)

    df_raw = pd.read_csv(self.fp_uf_raw)

    count_right_move = 0
    for k in range (0,len(df_score)-1,2):
        x1 = df_score['AbNatiV selected portion score'][k] # Native
        x2 = df_score['AbNatiV selected portion score'][k+1] # Grafted
        dx = x2 - x1
        
   
        y1 = df_raw['Kd (nM)'][k]
        y2 = df_raw['Kd (nM)'][k+1]
        dy=  y2 - y1
        
        if dx < 0:
            count_right_move += 1

    self.log('Correctly predicted UF-grafting', torch.tensor(count_right_move), on_epoch=True, logger=True)


  def on_validation_epoch_end(self) -> None:
    """
    Handles the end of a validation epoch.

    This method performs the following actions at the end of each validation epoch:
    1. Logs the current time and epoch number.
    2. Creates an output directory for the current epoch if it doesn't already exist.
    3. Logs and saves validation metrics to the output directory.
    4. Analyzes and saves validation scores stored in self.validation_step_outputs in a .pkl file.
       The saved data includes:
       {
           "accuracy": <accuracy_value>,
           "filtered_model_outputs": [
               {'loss_pbe': <value>, 'recon_error_pbe': <value>, 'perplexity': <value>},
               ...
           ]
       }
    5. Scores PSSM (Position-Specific Scoring Matrix) using the current model and saves the results.
    6. Runs ROC (Receiver Operating Characteristic) and PR (Precision-Recall) analyses between validation scores and PSSM scores, and saves the resulting plots.
    7. Clears the validation step outputs to prepare for the next epoch.
    """
    current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    logging.info(f"Validation Epoch {self.current_epoch} finished at {current_time}")

    output_dir = f"analysis/{self.run_name}/{self.current_epoch}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
  
    self.log_and_save_validation_metrics(output_dir)
    self.save_scores_and_process_pssm(output_dir)

    if self.fp_al_uf:
      self.run_uf_grafting(output_dir)

    self.validation_step_outputs = []

    return


