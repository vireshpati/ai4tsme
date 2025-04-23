# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoderModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model('transformer_sc_raw')
class TransformerSCRawModel(FairseqEncoderModel):
    """
    Class for training a transformer for SC10 speech command classification task.
    """
    def __init__(self, args, encoder, task):
        super().__init__(encoder)
        self.encoder = encoder
        self.args = args
        self._max_positions = args.max_positions
        self.sentence_out_dim = args.sentence_class_num
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.classifier = nn.ModuleList([])
        self.classifier.append(nn.Sequential(Linear(args.classifier_in_dim, args.classifier_out_dim),
                                             self.dropout_module))
        self.classifier.extend([
            nn.Sequential(Linear(args.classifier_out_dim, args.classifier_out_dim), self.dropout_module)
            for _ in range(args.classifier_layers - 1)
        ])
        self.classifier_activation = utils.get_activation_fn(args.classifier_activation_fn)
        self.sentence_projection_layer = Linear(
            args.classifier_out_dim,
            self.sentence_out_dim,
            bias=False
        )
        self.sen_rep_type = getattr(args, "sen_rep_type", "cls")

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D', help='dropout probability after activation in FFN')
        parser.add_argument('--feature-dropout', action='store_true', help='apply feature dropout')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--encoder-hidden-dim', type=int, metavar='N', help='encoder hidden dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')

        # Arguments related to input and output embeddings
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--max-positions', type=int, help='number of positional embeddings to learn')

        # Arguments related to sentence level prediction
        parser.add_argument('--sentence-class-num', type=int, metavar='N', help='number of classes for sentence task')
        parser.add_argument('--sent-loss', action='store_true', help='if set, calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--apply-bert-init', action='store_true', help='use custom param initialization for BERT')

        # misc params
        parser.add_argument('--activation-fn', choices=utils.get_available_activation_fns(), help='activation function to use')
        parser.add_argument('--classifier-activation-fn', choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for classifier layer.')
        parser.add_argument('--encoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')

        parser.add_argument('--sen-rep-type', choices=['cls', 'mp'], default='mp')

    def forward(self, sample):
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        sentence_rep = self.encoder(src_tokens, src_lengths)
        sentence_rep = sentence_rep[1]  # Get the sentence representation

        for layer in self.classifier:
            sentence_rep = self.classifier_activation(layer(sentence_rep))

        sentence_logits = self.sentence_projection_layer(sentence_rep)
        return {'encoder_out': sentence_logits}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self._max_positions

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = args.max_positions

        encoder = TransformerSCRawEncoder(args, task)
        return cls(args, encoder, task)

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output["encoder_out"]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and feed-forward network."""

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = nn.MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.normalize_before = args.encoder_normalize_before

    def forward(self, x, padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape (seq_len, batch, embed_dim)
            padding_mask (ByteTensor): binary tensor of shape (batch, seq_len)
                indicating which positions in the encoder output should be ignored

        Returns:
            encoded output of shape (seq_len, batch, embed_dim)
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, key_padding_mask=padding_mask)[0]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerSCRawEncoder(FairseqEncoder):
    """Transformer encoder for SC10 classification."""

    def __init__(self, args, task):
        super().__init__(None)
        self.args = args
        
        # For raw audio, we use a simple linear projection as the embedding
        self.embed_projection = nn.Linear(1, args.encoder_embed_dim)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        
        # Position embedding
        self.embed_positions = nn.Parameter(
            torch.Tensor(args.max_positions, args.encoder_embed_dim)
        )
        nn.init.normal_(self.embed_positions, mean=0, std=args.encoder_embed_dim ** -0.5)
        
        # Build the transformer layers manually instead of using TransformerSentenceEncoder
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args) for _ in range(args.encoder_layers)
        ])
        
        # Layer normalization applied before/after transformer layers
        self.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
        self.layer_norm = LayerNorm(args.encoder_embed_dim) if self.encoder_normalize_before else None
        self.final_layer_norm = None if self.encoder_normalize_before else LayerNorm(args.encoder_embed_dim)
        
        self.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        # B x T -> B x T x 1
        x = src_tokens.unsqueeze(-1)
        
        # B x T x 1 -> B x T x D
        x = self.embed_projection(x)
        
        # Apply position embeddings (reuse the first max_position embeddings for longer sequences)
        positions = torch.arange(src_tokens.size(1), device=src_tokens.device).unsqueeze(0)
        positions = positions % self.args.max_positions
        position_embeddings = F.embedding(positions, self.embed_positions)
        x = x + position_embeddings
        
        # Apply dropout
        x = self.dropout_module(x)
        
        # Create padding mask
        padding_mask = None
        if src_tokens.eq(0).any():  # Assuming 0 is the padding index
            padding_mask = src_tokens.eq(0)
        
        # B x T x D -> T x B x D
        x = x.transpose(0, 1)
        
        # Apply transformer layers
        inner_states = [x]
        for layer in self.layers:
            x = layer(x, padding_mask)
            inner_states.append(x)
            
        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)
        
        # Get sentence representation based on type
        if self.sen_rep_type == 'mp':
            # Mean pooling over the sequence length
            if src_lengths is not None:
                # Create mask for proper averaging
                mask = (torch.arange(x.size(0), device=x.device)
                        .unsqueeze(1) < src_lengths.unsqueeze(0)).t().float()
                mask = mask.unsqueeze(-1)  # T x B x 1
                # Apply mask and average
                sentence_rep = (x * mask).sum(dim=0) / src_lengths.unsqueeze(1).float()
            else:
                sentence_rep = x.mean(dim=0)
        else:
            # Use first token as representation (like CLS token)
            sentence_rep = x[0]
        
        return inner_states, sentence_rep


@register_model_architecture('transformer_sc_raw', 'transformer_sc_raw')
def base_architecture(args):
    args.apply_bert_init = getattr(args, 'apply_bert_init', False)
    
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 60)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 240)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    
    args.dropout = getattr(args, 'dropout', 0.0)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.feature_dropout = getattr(args, 'feature_dropout', False)
    
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    
    args.classifier_layers = getattr(args, 'classifier_layers', 1)
    args.classifier_out_dim = getattr(args, 'classifier_out_dim', 2 * args.encoder_embed_dim)
    args.sentence_class_num = getattr(args, 'sentence_class_num', 10)
    args.classifier_activation_fn = getattr(args, 'classifier_activation_fn', 'gelu')
    args.classifier_in_dim = getattr(args, "classifier_in_dim", args.encoder_embed_dim)
    args.sent_loss = getattr(args, 'sent_loss', True)
    
    args.max_positions = getattr(args, 'max_positions', 16000)
    args.sen_rep_type = getattr(args, 'sen_rep_type', 'mp')


@register_model_architecture('transformer_sc_raw', 'transformer_sc_raw_base')
def transformer_sc_raw_base(args):
    base_architecture(args)


@register_model_architecture('transformer_sc_raw', 'transformer_sc_raw_big')
def transformer_sc_raw_big(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 72)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 288)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 6)
    base_architecture(args)


@register_model_architecture('transformer_sc_raw', 'transformer_sc_raw_small')
def transformer_sc_raw_small(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 48)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 192)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    base_architecture(args)