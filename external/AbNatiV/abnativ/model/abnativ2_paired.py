# (c) 2023 Sormannilab and Aubin Ramon
#
# AbNatiV model, Pytorch version
#
# ============================================================================
# fmt: off

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
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix
import time

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
from .analysis_functions import calculate_pssm_aucs_paired_with_mismatch, get_abnativ_scores_per_batch, evaluate_paired_pssm_with_mismatch, evaluate_pairing_pred_with_mismatch

def binary_focal_loss(probs: torch.Tensor, targets: torch.Tensor, gamma=2.0):
    """
    Focal loss for probabilities.
    """
    bce_loss = nn.BCELoss(reduction='none')
    loss = bce_loss(probs, targets)

    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma
    focal_loss = focal_weight * loss
    return focal_loss


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


class AdaptiveLayerNorm(nn.Module):
    """
    Implements adaptive layer normalisation similar to AlphaFold3 (algorithm 26),
    which performs layer normalisation on the primary tensor conditional on the
    secondary tensor.

    The key difference with the AlphaFold implementation is that the secondary tensor is
    aggregated via a mean over the token dimension before projecting, to allow for different
    numbers of tokens in the primary and secondary tensors.
    """

    def __init__(self, embed_dim: int):
        """
        :param embed_dim: The number of features in the input tensors.
        """
        super().__init__()

        self.ln_no_affine = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.ln_no_bias = nn.LayerNorm(embed_dim, bias=False)

        self.linear_s = nn.Linear(embed_dim, embed_dim)
        self.linear_no_bias_s = nn.Linear(embed_dim, embed_dim, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        :param a: The input tensor to apply layer norm on.
        :param s: A secondary tensor for adaptively modulating the normalization. It is assumed that the
            token dimension of this tensor is the second-to-last dimension.
        :return: The normalized tensor.
        """

        a = self.ln_no_affine(a)
        s = torch.mean(s, dim=-2, keepdim=True)
        s = self.ln_no_bias(s)
        a = self.sigmoid(self.linear_s(s)) * a + self.linear_no_bias_s(s)

        return a


class GatedCrossAttention(nn.Module):
    """
    Cross-attention layer for heavy/light chain embedding interactions. This layer uses
    an adaptive layer norm and a gating mechanism to modulate the attention weights obtained
    from the cross attention mechanism.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.1,
        gate_bias_value: float = 0.0,
    ):
        """
        :param embed_dim: The dimensionality of the input/output embeddings.
        :param num_heads: The number of attention heads to use. The embedding dimension :code:`embed_dim`
            must be divisible by this number.
        :param dropout_p: The dropout probability for the attention weights.
        :param gate_bias_value: The initialisation value for the gate bias. This can be initialised to a
            large negative value to prevent any cross-attention modulation at the start of training.
        """

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.layer_norm_heavy = AdaptiveLayerNorm(embed_dim)
        self.layer_norm_light = AdaptiveLayerNorm(embed_dim)

        self.q_heavy = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_light = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_light = nn.Linear(embed_dim, embed_dim, bias=False)

        self.q_light = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_heavy = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_heavy = nn.Linear(embed_dim, embed_dim, bias=False)

        self.linear_g_heavy = nn.Linear(embed_dim, embed_dim)
        self.linear_g_light = nn.Linear(embed_dim, embed_dim)

        self.linear_g_heavy.bias.data.fill_(gate_bias_value)
        self.linear_g_light.bias.data.fill_(gate_bias_value)

        self.sigmoid = nn.Sigmoid()

        self.output_heavy = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout_p)
        )
        self.output_light = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout_p)
        )

    def forward(
        self, heavy_repr: torch.Tensor, light_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on a pair of heavy and light chain embeddings.

        :param heavy_repr: :code:`(batch_size, vh_seq_len, embed_dim)` tensor of heavy chain embeddings.
        :param light_repr: :code:`(batch_size, vl_seq_len, embed_dim)` tensor of light chain embeddings.
        :return: Tuple of updated heavy and light chain embeddings, same shape as the inputs.
        """

        heavy_repr = self.layer_norm_heavy(heavy_repr, light_repr)
        light_repr = self.layer_norm_light(light_repr, heavy_repr)

        # Heavy to Light cross-attention
        q_heavy = rearrange(
            self.q_heavy(heavy_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        k_light = rearrange(
            self.k_light(light_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        v_light = rearrange(
            self.v_light(light_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        g_heavy = rearrange(
            self.sigmoid(self.linear_g_heavy(heavy_repr)),
            "b i (h d) -> b h i d",
            h=self.num_heads,
        )

        attn_output_heavy = (
            F.scaled_dot_product_attention(q_heavy, k_light, v_light) * g_heavy
        )
        attended_heavy = rearrange(attn_output_heavy, "b h i d -> b i (h d)")

        # Light to Heavy cross-attention
        q_light = rearrange(
            self.q_light(light_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        k_heavy = rearrange(
            self.k_heavy(heavy_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        v_heavy = rearrange(
            self.v_heavy(heavy_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        g_light = rearrange(
            self.sigmoid(self.linear_g_light(light_repr)),
            "b i (h d) -> b h i d",
            h=self.num_heads,
        )

        attn_output_light = (
            F.scaled_dot_product_attention(q_light, k_heavy, v_heavy) * g_light
        )
        attended_light = rearrange(attn_output_light, "b h i d -> b i (h d)")

        # Output projections
        updated_heavy = self.output_heavy(attended_heavy)
        updated_light = self.output_light(attended_light)

        return updated_heavy, updated_light
    

class SwiGLU(nn.Module):
    """A swish-gated linear unit (SwiGLU)."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        """
        :param input_dim: The dimension of the input tensor.
        :param output_dim: The dimension of the output tensor.
        """
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, output_dim, bias=bias)
        self.linear_2 = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single tensor of embeddings.

        :param x: :code:`(..., input_dim)` tensor of embeddings.
        :return: :code:`(..., output_dim)` tensor of updated embeddings.
        """
        a = self.linear_1(x)
        b = self.linear_2(x)
        x = F.silu(a) * b

        return x


class ConditionedSwiGLUTransition(nn.Module):
    """
    A conditional SwiGLU-activated transition block. This implements Algorithm 25 from AlphaFold3,
    with the addition of dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout_p: float = 0.1,
        gate_bias_value: float = -2.0,
    ):
        """
        :param embed_dim: Dimension of the input embeddings.
        :param dropout_p: Dropout probability for the transition layer.
        :param gate_bias_value: The initialisation value for the gate bias. This can be initialised to a
            large negative value to prevent any cross-attention modulation at the start of training.
        """
        super().__init__()

        self.layers = nn.Sequential(
            SwiGLU(embed_dim, hidden_dim),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.norm = AdaptiveLayerNorm(embed_dim)

        self.linear_gate = nn.Linear(embed_dim, embed_dim)
        self.linear_gate.bias.data.fill_(gate_bias_value)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to transform an input tensor :code:`x` conditional on a secondary tensor :code:`y`.

        :param x: Input tensor to be transformed.
        :param y: Tensor containing conditioning information.
        :return: The transformed tensor.
        """
        x = self.norm(x,y)
        out = self.layers(x)
        gate = torch.sigmoid(self.linear_gate(y))

        return out * gate


class CrossAttentionBlock(nn.Module):
  '''CrossAttention Transformer Block'''
  def __init__(self, d_embedding, num_heads, d_ff, dropout, 
               gate_bias_value: float = -2.0):
    super(CrossAttentionBlock, self).__init__()

    self.cross_attention = GatedCrossAttention(
            embed_dim=d_embedding,
            num_heads=num_heads,
            dropout_p=dropout,
            gate_bias_value=gate_bias_value,
        )

    self.dropout = nn.Dropout(dropout)

    self.transition_heavy = ConditionedSwiGLUTransition(
            embed_dim=d_embedding, hidden_dim=d_ff,
            dropout_p=dropout, gate_bias_value=gate_bias_value
        )
    
    self.transition_light = ConditionedSwiGLUTransition(
            embed_dim=d_embedding, hidden_dim=d_ff,
            dropout_p=dropout, gate_bias_value=gate_bias_value
        )

  def forward(self, vh_x, vl_x) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
      vh_x: Tensor, shape [batch_size, l_red, d_embedding]
      vh_y: Tensor, shape [batch_size, l_red, d_embedding]
    """

    # Attention
    vh_output, vl_output = self.cross_attention(vh_x, vl_x)  # (batch_size, l_red, d_embedding)
    vh_x = vh_x + self.dropout(vh_output)
    vl_x = vl_x + self.dropout(vl_output)

    # SWIGLU
    vh_x = vh_x + self.transition_heavy(vh_x, vl_x) # (batch_size, input_seq_len, d_embedding) + residual 
    vl_x = vl_x + self.transition_light(vl_x, vh_x)

    return vh_x, vl_x


class Encoder(nn.Module):
  def __init__(self, d_embedding, kernel, stride, num_heads, num_mha_layers, d_ff,
              length_seq, alphabet_size, dropout=0):
    super(Encoder, self).__init__()

    self.num_mha_layers = num_mha_layers
    self.d_embedding = d_embedding

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

  

class PairedEncoder(nn.Module):
  '''A paired encoder module with cross-attention between two 
  transformers to capture cross-correlation between the heavy and light chain.
  
  The paired encoder does not CNN1 embed anymore. It takes as inputs the output
  of the individual encoders (so post CNN1)'''

  def __init__(self, d_embedding, num_heads, d_ff,
              heavy_encoder: Encoder, light_encoder: Encoder, 
              freeze_heavy: bool=False, freeze_light: bool=False, 
              dropout=0, gate_bias_value:float=-2.0):
    
    super(PairedEncoder, self).__init__()

    if heavy_encoder.num_mha_layers != light_encoder.num_mha_layers:
            raise ValueError(
                "Heavy and light networks must have the same number of layers."
            )
    
    self.num_layers = heavy_encoder.num_mha_layers
    self.heavy_network = heavy_encoder
    self.light_network = light_encoder

    if freeze_heavy:
       for param in self.heavy_network.parameters():
          param.requires_grad = False
    if freeze_light:
       for param in self.light_network.parameters():
          param.requires_grad = False

    # Projections before cross attention
    self.vh_pre_projections = nn.ModuleList(
        [
            nn.Linear(heavy_encoder.d_embedding, d_embedding, bias=False)
            for _ in range(self.num_layers)
        ]
    )
    self.vl_pre_projections = nn.ModuleList(
        [
            nn.Linear(light_encoder.d_embedding, d_embedding, bias=False)
            for _ in range(self.num_layers)
        ]
      )

    # MHA blocks
    self.cross_attention_blocks = nn.ModuleList([CrossAttentionBlock(d_embedding, num_heads, d_ff, dropout, gate_bias_value)
                       for _ in range(self.num_layers)])
    
    
    # Projections after cross attention
    self.vh_post_projections = nn.ModuleList(
        [
            nn.Linear(d_embedding, heavy_encoder.d_embedding, bias=False)
            for _ in range(self.num_layers)
        ]
    )
    self.vl_post_projections = nn.ModuleList(
        [
            nn.Linear(d_embedding, light_encoder.d_embedding, bias=False)
            for _ in range(self.num_layers)
        ]
    )

  def forward(self, vh_x, vl_x) -> torch.Tensor: 
    """
    Args:
      vh_x: Tensor, shape [batch_size, input_seq_len, alphabet_size] representing heavy sequence
      vl_x: Tensor, shape [batch_size, input_seq_len, alphabet_size] representing light sequence
    """

    # Embedding
    vh_repr = self.heavy_network.cnn_embedding(vh_x) 
    vh_repr = self.heavy_network.en_dropout(vh_repr) 

    vl_repr = self.light_network.cnn_embedding(vl_x) 
    vl_repr = self.light_network.en_dropout(vl_repr) 

    # MHA blocks
    for i in range(self.num_layers):
        # Self-attention on each chain
        vh_repr = self.heavy_network.en_MHA_blocks[i](vh_repr)
        vl_repr = self.light_network.en_MHA_blocks[i](vl_repr)

        # Project to shared embedding space
        vh_repr_cross = self.vh_pre_projections[i](vh_repr)
        vl_repr_cross = self.vl_pre_projections[i](vl_repr)

        # Cross-attention between heavy and light representations
        vh_repr_cross, vl_repr_cross = self.cross_attention_blocks[i](
            vh_repr_cross, vl_repr_cross
        )

        # Project back to original embedding spaces, with residual connection
        vh_repr = vh_repr + self.vh_post_projections[i](vh_repr_cross)
        vl_repr = vl_repr + self.vl_post_projections[i](vl_repr_cross)
    
    return vh_repr, vl_repr



class Decoder(nn.Module):
  def __init__(self, d_embedding, kernel, stride, num_heads, num_mha_layers, d_ff,
                  length_seq, alphabet_size, dropout=0):
    super(Decoder, self).__init__()

    self.num_mha_layers = num_mha_layers
    self.d_embedding = d_embedding

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
      
    # CNN reconstruction 
    z = self.cnn_reconstruction(z) # (batch_size, input_seq_len, alphabet_size)
    z_recon = F.softmax(z, dim=-1)

    return z_recon




class PairedDecoder(nn.Module):
  '''A paired decoder module with cross-attention between two 
  transformers to capture cross-correlation between the heavy and light chain.
  
  The paired decoder does not CNN1 embed anymore. It takes as inputs the output
  of the individual decoders (so post CNN1)'''

  def __init__(self, d_embedding, num_heads, d_ff,
              heavy_decoder: Decoder, light_decoder: Decoder, 
              freeze_heavy: bool=False, freeze_light: bool=False, 
              dropout=0, gate_bias_value:float=-2.0):
    
    super(PairedDecoder, self).__init__()

    if heavy_decoder.num_mha_layers != light_decoder.num_mha_layers:
            raise ValueError(
                "Heavy and light networks must have the same number of layers."
            )
    
    self.num_layers = heavy_decoder.num_mha_layers
    self.heavy_network = heavy_decoder
    self.light_network = light_decoder

    if freeze_heavy:
       for param in self.heavy_network.parameters():
          param.requires_grad = False
    if freeze_light:
       for param in self.light_network.parameters():
          param.requires_grad = False

    # Projections before cross attention
    self.vh_pre_projections = nn.ModuleList(
        [
            nn.Linear(heavy_decoder.d_embedding, d_embedding, bias=False)
            for _ in range(self.num_layers)
        ]
    )
    self.vl_pre_projections = nn.ModuleList(
        [
            nn.Linear(light_decoder.d_embedding, d_embedding, bias=False)
            for _ in range(self.num_layers)
        ]
      )

    # MHA blocks
    self.cross_attention_blocks = nn.ModuleList([CrossAttentionBlock(d_embedding, num_heads, d_ff, dropout, gate_bias_value)
                       for _ in range(self.num_layers)])
    
    
    # Projections after cross attention
    self.vh_post_projections = nn.ModuleList(
        [
            nn.Linear(d_embedding, heavy_decoder.d_embedding, bias=False)
            for _ in range(self.num_layers)
        ]
    )
    self.vl_post_projections = nn.ModuleList(
        [
            nn.Linear(d_embedding, light_decoder.d_embedding, bias=False)
            for _ in range(self.num_layers)
        ]
    )

  def forward(self, vh_x, vl_x) -> torch.Tensor: 
    """
    Args:
      vh_x: Tensor, shape [batch_size, input_seq_len, alphabet_size] representing heavy sequence
      vl_x: Tensor, shape [batch_size, input_seq_len, alphabet_size] representing light sequence
    """

    # Embedding
    vh_repr = self.heavy_network.de_dropout(vh_x)
    vl_repr = self.light_network.de_dropout(vl_x) 

    # MHA blocks
    for i in range(self.num_layers):
        # Self-attention on each chain
        vh_repr = self.heavy_network.de_MHA_blocks[i](vh_repr)
        vl_repr = self.light_network.de_MHA_blocks[i](vl_repr)

        # Project to shared embedding space
        vh_repr_cross = self.vh_pre_projections[i](vh_repr)
        vl_repr_cross = self.vl_pre_projections[i](vl_repr)

        # Cross-attention between heavy and light representations
        vh_repr_cross, vl_repr_cross = self.cross_attention_blocks[i](
            vh_repr_cross, vl_repr_cross
        )

        # Project back to original embedding spaces, with residual connection
        vh_repr = vh_repr + self.vh_post_projections[i](vh_repr_cross)
        vl_repr = vl_repr + self.vl_post_projections[i](vl_repr_cross)

    vh_pre_poj = vh_repr.clone()
    vl_pre_poj = vl_repr.clone()

    # CNN reconstruction 
    vh_repr = self.heavy_network.cnn_reconstruction(vh_repr) # (batch_size, input_seq_len, alphabet_size)
    vh_repr = F.softmax(vh_repr, dim=-1)

    vl_repr = self.light_network.cnn_reconstruction(vl_repr) # (batch_size, input_seq_len, alphabet_size)
    vl_repr = F.softmax(vl_repr, dim=-1)
    
    return vh_repr, vl_repr, vh_pre_poj, vl_pre_poj




class AbNatiV_Paired_Model(pl.LightningModule):
  def __init__(self, hparams: dict):
    super(AbNatiV_Paired_Model, self).__init__()

    # MODEL
    self.run_name = hparams["run_name"]

    # Paired encoder
    vh_encoder = Encoder(hparams['d_embedding'], hparams['kernel'], hparams['stride'], hparams['num_heads'], 
                            hparams['num_mha_layers'], hparams['d_ff'], hparams['length_seq'], 
                            hparams['alphabet_size'], dropout=hparams['drop'])
    
    vl_encoder = Encoder(hparams['d_embedding'], hparams['kernel'], hparams['stride'], hparams['num_heads'], 
                            hparams['num_mha_layers'], hparams['d_ff'], hparams['length_seq'], 
                            hparams['alphabet_size'], dropout=hparams['drop'])
  
    self.encoder = PairedEncoder(hparams['d_embedding_cross'],  hparams['num_heads_cross'], hparams['d_ff_cross'],
                                      vh_encoder, vl_encoder, hparams['freeze_heavy'], hparams['freeze_light'],
                                      hparams['drop'], hparams['gate_bias'])

    # Paired decoder
    vh_decoder = Decoder(hparams['d_embedding'], hparams['kernel'], hparams['stride'], hparams['num_heads'], 
                            hparams['num_mha_layers'], hparams['d_ff'], hparams['length_seq'], 
                            hparams['alphabet_size'], dropout=hparams['drop'])
    
    vl_decoder = Decoder(hparams['d_embedding'], hparams['kernel'], hparams['stride'], hparams['num_heads'], 
                            hparams['num_mha_layers'], hparams['d_ff'], hparams['length_seq'], 
                            hparams['alphabet_size'], dropout=hparams['drop'])
  
    self.decoder = PairedDecoder(hparams['d_embedding_cross'],  hparams['num_heads_cross'], hparams['d_ff_cross'],
                                      vh_decoder, vl_decoder, hparams['freeze_heavy'], hparams['freeze_light'],
                                      hparams['drop'], hparams['gate_bias'])
    

    # Codebooks
    self.vh_vqvae = VectorQuantize(
            dim=hparams['d_embedding'],
            codebook_size=hparams['num_embeddings'],
            codebook_dim=hparams['embedding_dim_code_book'],
            decay=hparams['decay'],
            kmeans_init=True,
            commitment_weight=hparams['commitment_cost']
            )
    
    self.vl_vqvae = VectorQuantize(
            dim=hparams['d_embedding'],
            codebook_size=hparams['num_embeddings'],
            codebook_dim=hparams['embedding_dim_code_book'],
            decay=hparams['decay'],
            kmeans_init=True,
            commitment_weight=hparams['commitment_cost']
            )
    
    # Freezing
    if hparams['freeze_heavy']:  # Freeze VH model if requested
        for param in self.vh_vqvae.parameters():
            param.requires_grad = False
        for param in self.encoder.heavy_network.parameters():
            param.requires_grad = False
        for param in self.decoder.heavy_network.parameters():
            param.requires_grad = False

    if hparams['freeze_light']:  # Freeze VL model if requested
        for param in self.vl_vqvae.parameters():
            param.requires_grad = False
        for param in self.encoder.light_network.parameters():
            param.requires_grad = False
        for param in self.decoder.light_network.parameters():
            param.requires_grad = False

    # Pairing prediction
    self.linear_heavy_ppred = nn.Sequential(
       nn.Linear(hparams['d_embedding'], hparams['d_ppred_lr'], bias=False),
        nn.LayerNorm(hparams['d_ppred_lr'])
    )
    
    self.linear_light_ppred =  nn.Sequential(
       nn.Linear(hparams['d_embedding'], hparams['d_ppred_lr'], bias=False),
        nn.LayerNorm(hparams['d_ppred_lr'])
    )
    
    self.lr = nn.Linear(hparams['d_ppred_lr']*vh_encoder.l_red, 1)
    # self.lr3 = nn.Linear(hparams['d_ppred_lr']*vh_encoder.l_red, 3)

    # TRAINING
    self.learning_rate = hparams['learning_rate']
    self.bce_loss = nn.BCELoss()

    # LOSS 
    if "loss" in hparams: self.loss = hparams['loss']
    else: self.loss = 'mse'
    if "gamma" in hparams: self.gamma = hparams['gamma']
    else: self.gamma = 1
    if "lambda" in hparams: self.lambda_ = hparams['lambda']
    else: self.lambda_ = 1

    # REAL-TIME ANALYSIS
    self.validation_step_outputs = []
    self.fp_diverse_test = hparams['fp_diverse_test']
    self.fp_pssm = hparams['fp_pssm']

    self.fp_test = hparams['fp_test']

    if 'fp_test_1l_perh_matching' in hparams:
        self.fp_test_1l_perh_matching = hparams['fp_test_1l_perh_matching']
        self.fp_test_1l_perh_binned = hparams['fp_test_1l_perh_binned']

        self.fp_test_50l_perh_matching = hparams['fp_test_50l_perh_matching']
        self.fp_test_50l_perh_binned = hparams['fp_test_50l_perh_binned']

        self.fp_test_mismatch = hparams['fp_test_mismatch']

        self.fp_mouse = hparams['fp_mouse']
        self.fp_rat = hparams['fp_rat']

        self.imgt_human = hparams['fp_imgt_human']
        self.imgt_non_human = hparams['fp_imgt_non_human']

        self.ada = hparams['fp_ada']

    if 'lr_head' in hparams:
       self.lr_head = hparams['lr_head']
    else:
       self.lr_head = False


    self.batch_size = hparams['batch_size']
    self.zeta = hparams['zeta']

    if "conservation_index" in hparams:
      self.conservation_index = hparams["conservation_index"]
    else: 
      self.conservation_index = None

    if "caa_freq" in hparams:
      self.aa_freq = hparams["aa_freq"]
    else: 
      self.aa_freq = None

    if "weight_decay" in hparams:
       self.weight_decay = hparams["weight_decay"]
    else:
       self.weight_decay = 0

    self.save_hyperparameters(hparams)
    self._pretrained_loaded = False

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
    
  def on_train_batch_start(self, batch, batch_idx):
        """Load pre-trained weights only at step 0"""

        self.batch_load_time = time.time() - self.load_start_time
        self.log(
            "train/load_time",
            self.batch_load_time,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.step_start_time = time.time()

        if self.global_step == 0:
            print('LOADING THE PRE-TRAINED MODELS')
            print(f"Epoch {self.current_epoch} | global_step: {self.global_step} | batch_idx: {batch_idx}")
            self._pretrained_loaded = True

            vh_ckpt = torch.load(self.hparams["pre_trained_vh_ckpt"], map_location=self.device)
            vl_ckpt = torch.load(self.hparams["pre_trained_vl_ckpt"], map_location=self.device)

            self.encoder.heavy_network.load_state_dict({key.replace('encoder.',''): value for key, value in vh_ckpt["state_dict"].items() if key.startswith("encoder.")})
            self.encoder.light_network.load_state_dict({key.replace('encoder.',''): value for key, value in vl_ckpt["state_dict"].items() if key.startswith("encoder.")})
            
            self.decoder.heavy_network.load_state_dict({key.replace('decoder.',''): value for key, value in vh_ckpt["state_dict"].items() if key.startswith("decoder.")})
            self.decoder.light_network.load_state_dict({key.replace('decoder.',''): value for key, value in vl_ckpt["state_dict"].items() if key.startswith("decoder.")})

            self.vh_vqvae.load_state_dict({key.replace('vqvae.',''): value for key, value in vh_ckpt["state_dict"].items() if key.startswith("vqvae.")})
            self.vl_vqvae.load_state_dict({key.replace('vqvae.',''): value for key, value in vl_ckpt["state_dict"].items() if key.startswith("vqvae.")})

            if self.hparams['freeze_heavy']:  # Freeze VH model if requested
                for param in self.vh_vqvae.parameters():
                    param.requires_grad = False
                for param in self.encoder.heavy_network.parameters():
                    param.requires_grad = False
                for param in self.decoder.heavy_network.parameters():
                    param.requires_grad = False

            if self.hparams['freeze_light']:  # Freeze VL model if requested
                for param in self.vl_vqvae.parameters():
                    param.requires_grad = False
                for param in self.encoder.light_network.parameters():
                    param.requires_grad = False
                for param in self.decoder.light_network.parameters():
                    param.requires_grad = False

  def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self.batch_step_time = time.time() - self.step_start_time
        self.log(
            "train/step_time",
            self.batch_step_time,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.load_start_time = time.time()


  def forward(self, data) -> dict:

    vh_inputs, vh_m_inputs = data[0]
    vl_inputs, vl_m_inputs = data[1]

    B = vh_inputs.size(0)

    vh_x, vl_x = self.encoder(vh_m_inputs, vl_m_inputs)
    vh_vq_outputs = self.vh_vqvae(vh_x)
    vl_vq_outputs = self.vl_vqvae(vl_x)
    vh_x_recon, vl_x_recon, vh_x_pre_proj, vl_x_pre_proj = self.decoder(vh_vq_outputs['quantize_projected_out'], vl_vq_outputs['quantize_projected_out'])

    # Loss computing 
    vh_recon_error_pposi = self.calculate_recon_error_pposi(vh_inputs, vh_x_recon)
    vh_recon_error_pbe = torch.mean(vh_recon_error_pposi, dim=1)
    vh_lambda_vq_output = self.lambda_ * vh_vq_outputs['loss_vq_commit_pbe']
    vh_loss_pbe = torch.add(vh_recon_error_pbe, vh_lambda_vq_output)

    vl_recon_error_pposi = self.calculate_recon_error_pposi(vl_inputs, vl_x_recon)
    vl_recon_error_pbe = torch.mean(vl_recon_error_pposi, dim=1)
    vl_lambda_vq_output = self.lambda_ * vl_vq_outputs['loss_vq_commit_pbe']
    vl_loss_pbe = torch.add(vl_recon_error_pbe, vl_lambda_vq_output)

    inputs = torch.cat([vh_inputs,vl_inputs], dim=-2)
    x_recon = torch.cat([vh_x_recon,vl_x_recon], dim=-2)
    recon_error_pposi = torch.cat([vh_recon_error_pposi,vl_recon_error_pposi], dim=-1)
    loss_pbe = (vh_loss_pbe + vl_loss_pbe)/2
    recon_error_pbe = (vh_recon_error_pbe + vl_recon_error_pbe)/2

    mse_pposi = self.calculate_MSE_recon_error_pposi(inputs, x_recon)
    
    # Pairing prediction (positive)
    dot_paired = torch.flatten(
        self.linear_heavy_ppred(vh_x_pre_proj) * self.linear_light_ppred(vl_x_pre_proj),
        start_dim=1
    )
    paired_pred = torch.sigmoid(self.lr(dot_paired))
    target_paired = torch.ones((B, 1), device=paired_pred.device)

  
    if self.training:

        # Shuffling lights for contrastive learning 
        shuffled_indices = torch.randperm(B)
        shuffled_vl_m_inputs = vl_m_inputs[shuffled_indices]

        from_shuff_vh_x, shuffled_vl_x = self.encoder(vh_m_inputs, shuffled_vl_m_inputs)
        from_shuff_vh_vq_outputs = self.vh_vqvae(from_shuff_vh_x)
        shuffled_vl_vq_outputs = self.vl_vqvae(shuffled_vl_x)
        from_shuff_vh_x_recon, shuffled_vl_x_recon, from_shuff_vh_pre_proj, shuffled_vl_pre_proj = self.decoder(
            from_shuff_vh_vq_outputs['quantize_projected_out'],
            shuffled_vl_vq_outputs['quantize_projected_out']
        )

        # Shuffled negative prediction
        dot_NOT_paired = torch.flatten(
            self.linear_heavy_ppred(from_shuff_vh_pre_proj) * self.linear_light_ppred(shuffled_vl_pre_proj),
            start_dim=1
        )
        NOT_paired_pred = torch.sigmoid(self.lr(dot_NOT_paired))
        target_NOT_paired = torch.zeros((B, 1), device=paired_pred.device)

        # Humatch mismatch prediction
        vh_mismatch_inputs, vh_mismatch_m_inputs = data[2]
        vl_mismatch_inputs, vl_mismatch_m_inputs = data[3]
        
        from_mismatch_vh_x, mismatch_vl_x = self.encoder(vh_mismatch_m_inputs, vl_mismatch_m_inputs)
        from_mismatch_vh_vq_outputs = self.vh_vqvae(from_mismatch_vh_x)
        mismatch_vl_vq_outputs = self.vl_vqvae(mismatch_vl_x)
        from_mismatch_vh_x_recon, mismatch_vl_x_recon, from_mismatch_vh_pre_proj, mismatch_vl_pre_proj = self.decoder(
            from_mismatch_vh_vq_outputs['quantize_projected_out'],
            mismatch_vl_vq_outputs['quantize_projected_out']
        )

        dot_mismatch_paired = torch.flatten(
            self.linear_heavy_ppred(from_mismatch_vh_pre_proj) * self.linear_light_ppred(mismatch_vl_pre_proj),
            start_dim=1
        )
        mismatch_pred = torch.sigmoid(self.lr(dot_mismatch_paired))
        target_mismatch = torch.zeros((B, 1), device=paired_pred.device)

        # weighted BCE losses (if focal, use binary_focal_loss())
        pos_loss = self.bce_loss(paired_pred, target_paired)
        shuf_loss = self.bce_loss(NOT_paired_pred, target_NOT_paired)
        mismatch_loss = self.bce_loss(mismatch_pred, target_mismatch)

        ppred_loss = (
            self.hparams["weight_positive"] * pos_loss.mean()
            + self.hparams["weight_shuffled"] * shuf_loss.mean()
            + self.hparams["weight_mismatch"] * mismatch_loss.mean()
        ) / (self.hparams["weight_positive"] + self.hparams["weight_shuffled"] + self.hparams["weight_mismatch"])


        return {
            'vh_inputs': vh_inputs, # (batch_size, input_seq_len, alphabet_size)
            'vh_x_recon': vh_x_recon, # (batch_size, input_seq_len, alphabet_size)
            'vh_recon_error_pposi': vh_recon_error_pposi, # (batch_size, input_seq_len)
            'vh_recon_error_pbe': vh_recon_error_pbe, # (batch_size)
            'vh_loss_pbe': vh_loss_pbe, # (batch_size)
            **{f"vh_{key}": value for key, value in vh_vq_outputs.items()},

            'vl_inputs': vl_inputs, # (batch_size, input_seq_len, alphabet_size)
            'vl_x_recon': vl_x_recon, # (batch_size, input_seq_len, alphabet_size)
            'vl_recon_error_pposi': vl_recon_error_pposi, # (batch_size, input_seq_len)
            'vl_recon_error_pbe': vl_recon_error_pbe, # (batch_size)
            'vl_loss_pbe': vl_loss_pbe, # (batch_size)
            **{f"vl_{key}": value for key, value in vl_vq_outputs.items()},

            'inputs': inputs,
            'x_recon': x_recon, # (batch_size, input_seq_len, alphabet_size)
            'mse_pposi': mse_pposi, # (batch_size, input_seq_len)
            'recon_error_pposi': recon_error_pposi, # (batch_size, input_seq_len)
            'recon_error_pbe': recon_error_pbe,
            'loss_pbe': loss_pbe,

            'pairing_pred': paired_pred,
            'ppred_loss': ppred_loss,

            'pos_loss': pos_loss.mean(),
            'shuf_loss': shuf_loss.mean(),
            'mismatch_loss': mismatch_loss.mean(),
        }
        
    else: # Only for deployable version
       
       ppred_loss = torch.Tensor([0])

       return {
            'vh_inputs': vh_inputs, # (batch_size, input_seq_len, alphabet_size)
            'vh_x_recon': vh_x_recon, # (batch_size, input_seq_len, alphabet_size)
            'vh_recon_error_pposi': vh_recon_error_pposi, # (batch_size, input_seq_len)
            'vh_recon_error_pbe': vh_recon_error_pbe, # (batch_size)
            'vh_loss_pbe': vh_loss_pbe, # (batch_size)
            **{f"vh_{key}": value for key, value in vh_vq_outputs.items()},

            'vl_inputs': vl_inputs, # (batch_size, input_seq_len, alphabet_size)
            'vl_x_recon': vl_x_recon, # (batch_size, input_seq_len, alphabet_size)
            'vl_recon_error_pposi': vl_recon_error_pposi, # (batch_size, input_seq_len)
            'vl_recon_error_pbe': vl_recon_error_pbe, # (batch_size)
            'vl_loss_pbe': vl_loss_pbe, # (batch_size)
            **{f"vl_{key}": value for key, value in vl_vq_outputs.items()},

            'inputs': inputs,
            'x_recon': x_recon, # (batch_size, input_seq_len, alphabet_size)
            'mse_pposi': mse_pposi, # (batch_size, input_seq_len)
            'recon_error_pposi': recon_error_pposi, # (batch_size, input_seq_len)
            'recon_error_pbe': recon_error_pbe,
            'loss_pbe': loss_pbe,

            'pairing_pred': paired_pred,
            'ppred_loss': ppred_loss,

            'vh_x_encoder_pre_vq': vh_x,
            'vl_x_encoder_pre_vq': vl_x,
            'vh_quantize_projected_out': vh_vq_outputs['quantize_projected_out'],
            'vl_quantize_projected_out': vl_vq_outputs['quantize_projected_out'],
            'vh_x_decoder_pre_proj': vh_x_pre_proj,
            'vl_x_decoder_pre_proj': vl_x_pre_proj,
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

    if self.lr_head is False:
        optim_groups = list(self.encoder.parameters()) + \
                        list(self.decoder.parameters()) + \
                        list(self.vh_vqvae.parameters()) + \
                        list(self.vl_vqvae.parameters()) + \
                        \
                        list(self.linear_heavy_ppred.parameters()) + \
                        list(self.linear_light_ppred.parameters()) + \
                        list(self.lr.parameters())

        return torch.optim.AdamW(optim_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
    
    else:
        head = list(self.linear_heavy_ppred.parameters()) + \
            list(self.linear_light_ppred.parameters()) + \
            list(self.lr.parameters())

        backbone = list(self.encoder.parameters()) + \
                list(self.decoder.parameters()) + \
                list(self.vh_vqvae.parameters()) + \
                list(self.vl_vqvae.parameters())

        return torch.optim.AdamW(
            [
                {"params": backbone, "lr": self.learning_rate, "weight_decay": self.weight_decay},
                {"params": head,     "lr": self.lr_head, "weight_decay": self.weight_decay},
            ]
        )

  def training_step(self, batch, batch_idx) -> float:
    if batch_idx % 1000 == 0:
      current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
      logging.info(f"Training on batch {batch_idx} at {current_time}")
    
    vqvae_output = self(batch)

    loss_vqvae = torch.mean(vqvae_output['loss_pbe'])
    self.log("train_loss_vqvae", loss_vqvae, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    loss_vqvae_vh = torch.mean(vqvae_output['vh_loss_pbe'])
    self.log("train_loss_vqvae_vh", loss_vqvae_vh, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    loss_vqvae_vl = torch.mean(vqvae_output['vl_loss_pbe'])
    self.log("train_loss_vqvae_vl", loss_vqvae_vl, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    loss_vq_commit_vh = torch.mean(vqvae_output['vh_loss_vq_commit_pbe'])
    self.log("train_loss_vq_commit_vh", loss_vq_commit_vh, on_step=True,on_epoch=True,prog_bar=True, logger=True)
    loss_vq_commit_vl = torch.mean(vqvae_output['vl_loss_vq_commit_pbe'])
    self.log("train_loss_vq_commit_vl", loss_vq_commit_vl, on_step=True,on_epoch=True,prog_bar=True, logger=True)

    nmse_accuracy_vh = torch.mean(vqvae_output['vh_recon_error_pbe'])
    self.log("train_loss_nmse_recons_vh", nmse_accuracy_vh, on_step=True, on_epoch=True,prog_bar=True, logger=True)
    nmse_accuracy_vl = torch.mean(vqvae_output['vl_recon_error_pbe'])
    self.log("train_loss_nmse_recons_vl", nmse_accuracy_vl, on_step=True, on_epoch=True,prog_bar=True, logger=True)

    vh_perplexity = vqvae_output['vh_perplexity']
    self.log("train_perplexity_vh", vh_perplexity, on_step=True, on_epoch=True,prog_bar=True, logger=True)
    vl_perplexity = vqvae_output['vl_perplexity']
    self.log("train_perplexity_vl", vl_perplexity, on_step=True, on_epoch=True,prog_bar=True, logger=True)

    ppred_loss = vqvae_output['ppred_loss']
    self.log("ppred_loss", ppred_loss, on_step=True, on_epoch=True,prog_bar=True, logger=True)

    total_loss = loss_vqvae + self.zeta * ppred_loss
    self.log("train_loss_vqvae_pairing", total_loss, on_step=True, on_epoch=True,prog_bar=True, logger=True)

    self.log("pos_loss", vqvae_output['pos_loss'], on_step=True, on_epoch=True,prog_bar=True, logger=True)
    self.log("shuf_loss", vqvae_output['shuf_loss'], on_step=True, on_epoch=True,prog_bar=True, logger=True)
    self.log("mismatch_loss", vqvae_output['mismatch_loss'], on_step=True, on_epoch=True,prog_bar=True, logger=True)

    self.log('pairing_pred', vqvae_output['pairing_pred'].mean(), on_step=True, on_epoch=True,prog_bar=True, logger=True)

    return total_loss

  def on_train_epoch_start(self) -> None:
    current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    logging.info(f"Training Epoch {self.current_epoch} started at {current_time}")
    self.load_start_time = time.time()
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

    mean_ppred_loss = torch.mean(model_output['ppred_loss'])
    mean_pairing_pred = torch.mean(model_output['pairing_pred'])

    mean_total_loss = mean_loss + self.zeta * mean_ppred_loss


    mean_output = {
      "mean_total_loss": mean_total_loss,
      "mean_loss": mean_loss,
      "mean_recon_error": mean_recon_error,
      "mean_accuracy": mean_accuracy,
      "mean_ppred_loss": mean_ppred_loss,
      "mean_pairing_pred": mean_pairing_pred
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

    val_mean_total_loss = torch.Tensor([out["mean_total_loss"] for out in model_outputs])
    total_val_mean_total_loss = torch.mean(val_mean_total_loss)
    self.log('val_total_loss', total_val_mean_total_loss, on_epoch=True, logger=True)

    val_losses = torch.Tensor([out["mean_loss"] for out in model_outputs])
    total_val_loss = torch.mean(val_losses)
    self.log('val_vqvae_loss', total_val_loss, on_epoch=True, logger=True)

    val_accuracies = torch.Tensor([out['mean_recon_error'] for out in model_outputs])
    total_val_accuracy = torch.mean(val_accuracies)
    self.log('val_nmse_accuracy', total_val_accuracy, on_epoch=True, logger=True)

    accuracies = torch.Tensor([out['mean_accuracy'] for out in model_outputs])
    total_val_accuracy = torch.mean(accuracies)
    self.log('val_accuracy', total_val_accuracy, on_epoch=True, logger=True)

    val_ppred_loss = torch.Tensor([out['mean_ppred_loss'] for out in model_outputs])
    total_val_ppred_loss = torch.mean(val_ppred_loss)
    self.log('val_ppred_loss', total_val_ppred_loss, on_epoch=True, logger=True)

    val_mean_pairing_pred = torch.Tensor([out['mean_pairing_pred'] for out in model_outputs])
    total_mean_pairing_pred = torch.mean(val_mean_pairing_pred)
    self.log('val_mean_pairing_pred', total_mean_pairing_pred, on_epoch=True, logger=True)

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
    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_epoch, 
      fp_pssm=self.fp_pssm, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

    self.log('PR Val-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_epoch, 
      fp_pssm=self.fp_mouse, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

    self.log('PR Val-Mouse', torch.tensor(pr_auc), on_epoch=True, logger=True)

    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_epoch, 
      fp_pssm=self.fp_rat, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

    self.log('PR Val-Rat', torch.tensor(pr_auc), on_epoch=True, logger=True)

    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_epoch, 
      fp_pssm=self.fp_diverse_test, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

    self.log('PR Val-Diverse', torch.tensor(pr_auc), on_epoch=True, logger=True)


    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_epoch, 
      fp_pssm=self.fp_test_mismatch, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

    self.log('PR Val-Mismatch', torch.tensor(pr_auc), on_epoch=True, logger=True)


    # IMGT TEST
    df_mean_imgt_human = evaluate_paired_pssm_with_mismatch(self, 'None', self.imgt_human, self.batch_size)

    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_imgt_human, 
      fp_pssm=self.imgt_non_human, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

    self.log('PR IMGT Human-Non-Human', torch.tensor(pr_auc), on_epoch=True, logger=True)

    # ADA TEST
    df_mean_ada = evaluate_paired_pssm_with_mismatch(self, 'None', self.ada, self.batch_size)
    ada_scores = df_mean_ada["AbNatiV None Score"]
    df_ada_immuno = pd.read_csv(self.ada)
    immunogenicities = df_ada_immuno['Immunogenicity']

    correlation, p_value = pearsonr(ada_scores, immunogenicities)

    self.log('ADA pearson correlation', torch.tensor(correlation), on_epoch=True, logger=True)


    ### DIVERSE vs others ###
    df_mean_diverse_test = evaluate_paired_pssm_with_mismatch(self, 'None', self.fp_diverse_test, self.batch_size)


    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_diverse_test, 
      fp_pssm=self.fp_pssm, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

    self.log('PR Diverse-PSSM', torch.tensor(pr_auc), on_epoch=True, logger=True)

    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_diverse_test, 
      fp_pssm=self.fp_mouse, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

    self.log('PR Diverse-Mouse', torch.tensor(pr_auc), on_epoch=True, logger=True)

    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
      model=self, 
      model_type='None', 
      df_mean_val=df_mean_diverse_test, 
      fp_pssm=self.fp_rat, 
      batch_size=self.batch_size, 
      output_dir=output_dir
    )

    self.log('PR Diverse-Rat', torch.tensor(pr_auc), on_epoch=True, logger=True)

    


    ## PAIRING PERFORMANCES OF THE PAIRING VQ-VAE RECONSRUCTION ##
    # Pairing info 
    scores_p_heavy_true = evaluate_paired_pssm_with_mismatch(self, 'None', self.fp_test_1l_perh_matching, self.batch_size, aho_posi_whole=range(1,150))["AbNatiV None Score"]
    scores_p_heavy_random = evaluate_paired_pssm_with_mismatch(self, 'None', self.fp_test_1l_perh_binned, self.batch_size, aho_posi_whole=range(1,150))["AbNatiV None Score"]
    c_correct_pairing = np.sum(np.array(scores_p_heavy_true)[:len(scores_p_heavy_random)] > np.array(scores_p_heavy_random))/len(scores_p_heavy_random)

    self.log('[VQ-VAE Reconstruction] Correct pairing prediction Test 1:1 (on 10000H) [Vgene BINNED]', torch.tensor(c_correct_pairing), on_epoch=True, logger=True)

    mean_diff = np.mean(np.abs(np.array(scores_p_heavy_true)[:len(scores_p_heavy_random)] - np.array(scores_p_heavy_random)))
    self.log('[VQ-VAE Reconstruction] Mean Abs Diff Paired-Unpaired', torch.tensor(mean_diff), on_epoch=True, logger=True)

    # Pairing info per 50 (binned)
    scores_p_heavy_random_50_matching = evaluate_paired_pssm_with_mismatch(self, 'None', self.fp_test_50l_perh_matching, self.batch_size, aho_posi_whole=range(1,150))["AbNatiV None Score"]
    scores_p_heavy_random_50 = evaluate_paired_pssm_with_mismatch(self, 'None', self.fp_test_50l_perh_binned, self.batch_size, aho_posi_whole=range(1,150))["AbNatiV None Score"]
    max_heavy_seqs = 1000

    scores_p_heavy_random_50_mean = list()
    for i in range(0, len(scores_p_heavy_random_50), 50):
        scores_p_heavy_random_50_mean.append(np.mean(scores_p_heavy_random_50[i:i+50]))

    c_correct_pairing = np.sum(np.array(scores_p_heavy_random_50_matching[:max_heavy_seqs]) > np.array(scores_p_heavy_random_50_mean))/len(scores_p_heavy_random_50_mean)
    self.log('[VQ-VAE Reconstruction] Correct pairing prediction Test 50L per H (on 1000H) [Vgene BINNED]', torch.tensor(c_correct_pairing), on_epoch=True, logger=True)

    ## PAIRED vs MISMATCH ##
    df_mean_paired_test = evaluate_paired_pssm_with_mismatch(self, 'None', self.fp_test, self.batch_size)
    df_mean_mismatch_test = evaluate_paired_pssm_with_mismatch(self, 'None', self.fp_test_mismatch, self.batch_size)
    paired_scores = df_mean_paired_test["AbNatiV None Score"]
    mismatch_scores = df_mean_mismatch_test["AbNatiV None Score"]
    mean_diff = np.mean(paired_scores) - np.mean(mismatch_scores)
    self.log("Score separation Paired-Mismatch (mean)", torch.tensor(mean_diff), on_epoch=True, logger=True)

    pr_auc, roc_auc = calculate_pssm_aucs_paired_with_mismatch(
        model=self,
        model_type='None',
        df_mean_val=df_mean_paired_test,
        fp_pssm=self.fp_test_mismatch,
        batch_size=self.batch_size,
        output_dir=output_dir
    )
    self.log("PR AUC Paired vs Mismatch", torch.tensor(pr_auc), on_epoch=True, logger=True)
    self.log("ROC AUC Paired vs Mismatch", torch.tensor(roc_auc), on_epoch=True, logger=True)


    ## PAIRING PERFORMANCES OF THE PAIRING PREDICTION HEAD ##
    # Pairing prediction head per 1 (binned)
    paired_pred_test_matching = evaluate_pairing_pred_with_mismatch(self, 'None', self.fp_test_1l_perh_matching, self.batch_size)
    paired_pred_shuffled = evaluate_pairing_pred_with_mismatch(self, 'None', self.fp_test_1l_perh_binned, self.batch_size)

    c_correct_pairing = np.sum(np.array(paired_pred_test_matching)[:len(paired_pred_shuffled)] > np.array(paired_pred_shuffled))/len(paired_pred_test_matching)
    self.log('[Prediction head] Correct pairing prediction Test 1:1 (on 10000H) [Vgene BINNED]', torch.tensor(c_correct_pairing), on_epoch=True, logger=True)

    # Pairing prediction head per 50 (binned)
    paired_pred_test_50matching = evaluate_pairing_pred_with_mismatch(self, 'None', self.fp_test_50l_perh_matching, self.batch_size)
    paired_pred_50_shuffled_binned = evaluate_pairing_pred_with_mismatch(self, 'None', self.fp_test_50l_perh_binned, self.batch_size)

    scores_paring_pred_heavy_random_50_mean_perbine = list()
    for i in range(0, len(paired_pred_50_shuffled_binned), 50):
        scores_paring_pred_heavy_random_50_mean_perbine.append(np.mean(paired_pred_50_shuffled_binned[i:i+50]))

    c_correct_pairing = np.sum(np.squeeze(np.array(paired_pred_test_50matching[:max_heavy_seqs])) > np.array(scores_paring_pred_heavy_random_50_mean_perbine))/len(scores_paring_pred_heavy_random_50_mean_perbine)
    self.log('[Prediction head] Correct pairing prediction Test 50L per H (on 1000H) [Vgene BINNED]', torch.tensor(c_correct_pairing), on_epoch=True, logger=True)

    # Compute confusion matrix
    label_pred_pairing_test_matching = list(np.array(paired_pred_test_matching) > 0.5)
    label_pred_pairing_test_shuffle = list(np.array(paired_pred_shuffled) > 0.5)

    tp, fn, fp, tn = confusion_matrix([True]*len(label_pred_pairing_test_matching)+[False]*len(label_pred_pairing_test_shuffle), 
                            label_pred_pairing_test_matching+label_pred_pairing_test_shuffle,
                            labels=[True,False]).ravel()
    
    self.log('[Prediction head] FN rate Test 1:1 (on 10000H) [Vgene BINNED]', torch.tensor(fn/len(label_pred_pairing_test_matching)), on_epoch=True, logger=True)
    self.log('[Prediction head] FP rate Test 1:1 (on 10000H) [Vgene BINNED]', torch.tensor(fp/len(label_pred_pairing_test_shuffle)), on_epoch=True, logger=True)



    ## ADD MISMATCH ##
    label_paired_pred_mismatch = evaluate_pairing_pred_with_mismatch(self, 'None', self.fp_test_mismatch, self.batch_size)
    nb_bellow_05_mismatch = sum(np.array(label_paired_pred_mismatch) < 0.5)

    self.log('[Prediction head] Ratio mismatch < 0.5 (on 50000H)', torch.tensor(nb_bellow_05_mismatch/len(label_paired_pred_mismatch)), on_epoch=True, logger=True)

    # Compute AUC Mismatch - Test 
    label_paired_pred_all_test = evaluate_pairing_pred_with_mismatch(self, 'None', self.fp_test, self.batch_size)
    
    combined_scores = pd.concat([pd.DataFrame(label_paired_pred_all_test), pd.DataFrame(label_paired_pred_mismatch)])
    true_labels = pd.Series([1] * len(label_paired_pred_all_test) + [0] * len(label_paired_pred_mismatch))
    precision, recall, thresholds = precision_recall_curve(
        true_labels, combined_scores
    )
    pr_auc = auc(recall, precision)

    self.log('[Prediction head] PR-AUC Test vs mismatch (on 50000H)', torch.tensor(pr_auc), on_epoch=True, logger=True)


    ## ADD PSSM ##
    label_paired_pred_pssm = evaluate_pairing_pred_with_mismatch(self, 'None', self.fp_pssm, self.batch_size)
    nb_bellow_05_pssm= sum(np.array(label_paired_pred_pssm) < 0.5)

    self.log('[Prediction head] Ratio PSSM < 0.5 (on 10000H)', torch.tensor(nb_bellow_05_pssm/len(label_paired_pred_pssm)), on_epoch=True, logger=True)

    ## ADD Mice ##
    label_paired_pred_mouse = evaluate_pairing_pred_with_mismatch(self, 'None', self.fp_mouse, self.batch_size)
    nb_bellow_05_mouse = sum(np.array(label_paired_pred_mouse) < 0.5)

    self.log('[Prediction head] Ratio Mouse < 0.5 (on 10000H)', torch.tensor(nb_bellow_05_mouse/len(label_paired_pred_mouse)), on_epoch=True, logger=True)

    ## ADD Rat ##
    label_paired_pred_rat = evaluate_pairing_pred_with_mismatch(self, 'None', self.fp_rat, self.batch_size)
    nb_bellow_05_rat = sum(np.array(label_paired_pred_rat) < 0.5)

    self.log('[Prediction head] Ratio Rat < 0.5 (on 10000H)', torch.tensor(nb_bellow_05_rat/len(label_paired_pred_rat)), on_epoch=True, logger=True)


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

    self.validation_step_outputs = []

    return

# fmt: on
