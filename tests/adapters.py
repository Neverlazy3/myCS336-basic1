'''
Author: harmoon 92302745+Neverlazy3@users.noreply.github.com
Date: 2025-10-11 15:27:36
LastEditors: harmoon 92302745+Neverlazy3@users.noreply.github.com
LastEditTime: 2025-10-30 20:00:14
FilePath: \assignment1-basics-main\tests\adapters.py
Description: 

Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
'''
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO, Iterable, Generator, Tuple , List, Dict

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import re
import regex
import collections as Collections
from collections import defaultdict, Counter


from cs336_basics.myAttenation import MultiHeadSelfAttention, MultiHeadSelfAttentionWithRoPe, ScaledDotProductAttention
from cs336_basics.myEmbedding import Embedding
from cs336_basics.myLinear import Linear
from cs336_basics.myRMSNorm import RMSNorm
from cs336_basics.myRoPE import RoPE
from cs336_basics.mySoftMax import softmax
from cs336_basics.mySwiGLU import SwiGLU
from cs336_basics.myTransformerBlock import transformerBlock
from cs336_basics.myTransformer_lm import Transformer_lm


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""     

# help function
def word_to_bytes_tuple(word: str):
    word_bytes_list = list(word.encode("utf-8"))
    # word.encode("utf-8")得到b'hello'，再转成list得到[104, 101, 108, 108, 111]
    bytes_list = [bytes([b]) for b in word_bytes_list]
    # []的作用是将其转化为数字，例如bytes(x)得到是纯字节串，[x]得到的是unicode数字，bytes([x])得到的是unicode对应的字符
    # bytes_list的最终结果是[b'h', b'e', b'l', b'l', b'o']
    # tuple(bytes_list)的最终结果是(b'h', b'e', b'l', b'l', b'o')
    return tuple(bytes_list)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(d_in, d_out, weights.device, weights.dtype)
    linear.load_state_dict({"weight": weights})
    return linear(in_features)
    # raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(vocab_size, d_model, weights.device, weights.dtype)
    embedding.load_state_dict({"weight": weights})
    return embedding(token_ids)
    # raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLU(d_ff=d_ff, d_modle=d_model)
    swiglu.load_state_dict({"w1_weight": w1_weight, "w2_weight": w2_weight, "w3_weight": w3_weight})
    
    return swiglu(in_features)
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scaled_dot_product_attention =  ScaledDotProductAttention()
    return scaled_dot_product_attention(Q, K, V, mask)
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_v d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multi_head_attention = MultiHeadSelfAttention(d_model, num_heads)
    return multi_head_attention(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features)
    
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    attention_rope = MultiHeadSelfAttentionWithRoPe(d_model, num_heads, max_seq_len, theta)
    return attention_rope(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features, token_positions)
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RoPE(theta, d_k, max_seq_len, in_query_or_key.device)
    return rope(in_query_or_key, token_positions)
    # raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    trans_block = transformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, in_features.device, in_features.dtype)
    trans_block.load_state_dict(weights)
    return trans_block(in_features, None)
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_lm = Transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, in_indices.device, in_indices.dtype)
    transformer_lm.load_state_dict(weights)
    return transformer_lm(in_indices, None)
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms_norm = RMSNorm(d_model, eps, weights.device, weights.dtype)
    rms_norm.load_state_dict({"weight" : weights})
    return rms_norm(in_features)
    # raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim)
    # raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    
    
    return tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    # raise NotImplementedError

class tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.byte2id = {v : k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []
        # breakpoint()
        for special_tokens_item in self.special_tokens:
            bytes_item = special_tokens_item.encode("utf-8")
            if bytes_item not in self.byte2id:
                new_id = len(self.vocab)
                token_byte = bytes_item
                self.byte2id[token_byte] = new_id
                self.vocab[new_id] = token_byte
    
    def encode(self, text: str) -> list[int]:
        tokens = []  # 最终存储编码后的token ID列表
        # 1. 对特殊token按长度倒序排序（避免短特殊token被长的误拆分，例如"<EOS>"不会被"<EO>"提前匹配）
        # 假设self.special_tokens = ["<EOS>", "<POS>"]（示例特殊token），排序后仍为["<EOS>", "<POS>"]（长度相同）
        sorted_special_token = sorted(self.special_tokens, key=len, reverse=True)
        
        # 2. 构建匹配特殊token的正则模式（用|拼接，且转义特殊字符）
        # 示例中pattern会被构建为："<EOS>|<POS>"（regex.escape确保特殊字符如<、>不被解析为正则语法）
        pattern = "|".join(map(regex.escape, sorted_special_token))
        # breakpoint()
        # 3. 按特殊token分割原始文本，同时保留特殊token本身
        if pattern:
            # 对示例文本 "hello world!<EOS>How are you<POS>" 分割后：
            # text_list = ["hello world!", "<EOS>", "How are you", "<POS>", ""]
            # （注：末尾空字符串是因为文本以特殊token结束，split会保留最后一个分隔符后的空内容）
            text_list = regex.split(f"({pattern})", text)
        else:
            # 若没有特殊token，直接将整个文本作为列表唯一元素
            text_list = [text]

        # breakpoint()
        # 4. 遍历分割后的片段，分别处理特殊token和普通文本
        for text_item in text_list:
            # 跳过空字符串（split可能产生，不影响实际内容）
            if not text_item:
                continue
            # 4.1 若当前片段是特殊token，直接获取其对应的ID
            if text_item in sorted_special_token:
                # 示例中：text_item为"<EOS>"时，编码为bytes "<EOS>".encode("utf-8")，
                # 通过self.byte2id（字节到ID的映射）获取其ID（假设为100），添加到tokens
                tokens.append(self.byte2id[text_item.encode("utf-8")])
            else:
                words = regex.findall(PAT, text_item)
                for word in words:
                    word_bytes = word.encode("utf-8")
                    word_tokens = bpe_encode(self.byte2id, word_bytes)
                    tokens.extend(word_tokens)
                    
        # 最终返回完整的token ID列表
        # 示例结果可能为：[10, 3, 20, 5, 100, 15, 8, 25, 101, ...]
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Generator[int, None, None]:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(self, tokens: list[int]) -> str:
        word_bytes = b"".join(self.vocab[token] for token in tokens)
        # print(f"word_bytes:{word_bytes}")
        return word_bytes.decode("utf-8", errors="replace")

def bpe_encode(merge_ranks: dict[bytes, int], word: bytes) -> list[int]:
        tokens = []  # 用于存储匹配到的文本片段（字符串形式）
        parts = [bytes([b]) for b in word]
        
        while True:
            min_idx = None
            min_rank = None
            
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = merge_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
        
            if min_rank is None:
                break
            assert min_idx is not None
            
            parts = parts[: min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2: ]
            
        tokens = [merge_ranks[part] for part in parts]
        return tokens

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """ 
    
    def get_counts(pre_tokens_cnt: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
        counts = defaultdict(int)
        for tokens, count in pre_tokens_cnt.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                counts[pair] += count
        return counts
    
    def merge(pre_tokens_cnt, pair_target):
        newTokens = []
        for token, count in pre_tokens_cnt.items():
            # i = 0
            # new_pre_token = []
            # while i < len(token):
            #     # 检查是否可以合并
            #     if i < len(token) - 1 and (token[i], token[i + 1]) == pair_target:
            #         new_pre_token.append(pair_target[0] + pair_target[1])  # 合并成一个新的 bytes
            #         i += 2
            #     else:
            #         new_pre_token.append(token[i])
            #         i += 1
            # new_pre_token = tuple(new_pre_token)
            # newTokens.append((token, new_pre_token, count))
            
            indices = [i for i in range(len(token) - 1) if (token[i], token[i + 1]) == pair_target]
            if indices:
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(pair_target[0] + pair_target[1])  # 合并成一个新的 bytes
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                newTokens.append((token, new_pre_token, count))
        return newTokens

    # 接下来的改进路径是降低时间复杂度！
    def encode(input_path, vocab_size, special_tokens):
        # -----------1.构建词汇表-----------
        ansVocabs = {i : bytes([i]) for i in range(256)}
        # vocabs是由“字词”->id组成的字典，ansVocabs是由id->“字词”组成的字典
        # 前者用于decode，后者用于返回函数符合要求的vocab

        special_tokens_bytes = [token.encode("utf-8") for token in special_tokens]
        # 假设special_tokens是"<EOS>"，那么special_tokens_bytes就是b"<EOS>"
        for token_bytes in special_tokens_bytes:
            # token_bytes示例： b"<EOS>"
            if token_bytes not in ansVocabs.values():
                ansVocabs[len(ansVocabs)] = token_bytes
        
        # -----------2.pre-tokenization-----------
        if isinstance(input_path, os.PathLike):
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = input_path

        pre_tokens_cnt = defaultdict(int)
        # 例如text是“Hello<EOS>world<PAD>test”
        #  special_tokens 是一个特殊符号列表，比如 ["<PAD>", "<EOS>", "<UNK>"]。
        # map(re.escape, special_tokens) 对每个特殊符号做正则转义，防止符号里有特殊字符影响分割。
        # "|".join(...) 把所有特殊符号用 | 连接，形成正则表达式，表示“遇到任意一个特殊符号都分割”。
        # re.split(...) 用这个正则表达式在 text 上分割，得到的 chunks 是原始文本被特殊符号分割后的所有片段。
        chunks = regex.split("|".join(map(regex.escape, special_tokens)), text)
        # 得到的chunks就是["Hello", "world", "test"]
        for chunk in chunks:
            for temp in regex.finditer(PAT, chunk):
                word = temp.group(0)
                # word是单词字符串,group()函数返回匹配的字符串
                # 这里假设word是"hello",用 to_bytes_tuple(word) 把它转成 UTF-8 字节序列(如 "Hello" → (b'H', b'e', b'l', b'l', b'o'))
                pre_tokens_cnt[word_to_bytes_tuple(word)] += 1


        # -----------3.BPE训练-----------
        ansMerges = []

        num_merges = vocab_size - 256 - len(special_tokens)# vocab_size是我们期望的词汇表库的大小，256是Unicode编码上限，还需要给特殊字符留位置
        if num_merges < 0:
            raise ValueError("vocab_size must be at least 256 + len(special_tokens)")
        
        # print(f"num merges: {num_merges}")
        print("encode Start------------------")
        for i in range(num_merges):
            counts = get_counts(pre_tokens_cnt)
            
            if not counts:
                print(f"Early stop at {i} merges (no more pairs)")
                break

            # 牛魔的搁这搞半天，还得考虑多个pair并列的情况，然后取字典序最大的那个，怪不得其他都对的上就数据的位置对不上
            max_count = max(counts.values())
            top_pairs = [pair for pair, count in counts.items() if count == max_count]
            top_pair = max(top_pairs)
            
            new_tokenId = len(ansVocabs)
            token_bytes = top_pair[0] + top_pair[1]
            
            # print(f"{top_pair[0]} {top_pair[1]} {new_tokenId - 255}")
            ansMerges.append((top_pair[0], top_pair[1]))
            ansVocabs[new_tokenId] = token_bytes


            for old_token, new_token, cnt in merge(pre_tokens_cnt, top_pair):
                # 例如old_token是[b'H', b'e', b'l', b'l', b'o']
                # 例如new_token是[b'He', b'l', b'l', b'o'],cnt则是old_token对应的频率
                pre_tokens_cnt[new_token] += cnt
                del pre_tokens_cnt[old_token]

        return ansVocabs, ansMerges
    
    return encode(input_path, vocab_size, special_tokens)
    # raise NotImplementedError
