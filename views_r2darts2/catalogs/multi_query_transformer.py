"""
Multi-Query Transformer model.

Fixes the single-query decoder limitation in Darts' stock TransformerModel.

Stock behaviour
---------------
``_create_transformer_inputs`` sets ``tgt = src[-1:]`` — one query token regardless
of output_chunk_length.  All output steps share the same cross-attention context
vector and are differentiated only by the linear decoder's weight slices.  This
provides no temporal inductive bias, causing the model to converge to a flat
per-series mean prediction.

Fix
---
Use ``tgt = src[-output_chunk_length:]`` so every output step gets its own
positional-encoded query token with independent cross-attention over the encoder
sequence.  The decoder is resized from
``d_model → target_length * output_size * nr_params`` (one-shot) to
``d_model → output_size * nr_params`` (per-step) so gradients flow distinctly
to each future step.
"""

import math

import torch
import torch.nn as nn

from darts.models.forecasting.transformer_model import TransformerModel, _TransformerModule
from darts.models.forecasting.pl_forecasting_module import io_processor


class _MultiQueryTransformerModule(_TransformerModule):
    """_TransformerModule with a per-step decoder and multi-step tgt queries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the fat one-shot decoder (d_model → target_length*output_size*nr_params)
        # with a per-step decoder (d_model → output_size*nr_params).
        d_model = self.decoder.in_features
        n_per_step = self.target_size * self.nr_params
        self.decoder = nn.Linear(d_model, n_per_step)

    def _create_transformer_inputs(self, data):
        src = data.permute(1, 0, 2)  # (input_chunk_length, batch, input_size)
        tgt = src[-self.target_length:, :, :]  # (output_chunk_length, batch, input_size)
        return src, tgt

    @io_processor
    def forward(self, x_in: tuple):
        data, *_ = x_in
        src, tgt = self._create_transformer_inputs(data)

        src = self.encoder(src) * math.sqrt(self.input_size)
        src = self.positional_encoding(src)

        tgt = self.encoder(tgt) * math.sqrt(self.input_size)
        tgt = self.positional_encoding(tgt)

        x = self.transformer(src=src, tgt=tgt)
        # x: (target_length, batch, d_model)

        out = self.decoder(x)
        # out: (target_length, batch, target_size * nr_params)

        predictions = out.permute(1, 0, 2)
        # predictions: (batch, target_length, target_size * nr_params)
        predictions = predictions.view(-1, self.target_length, self.target_size, self.nr_params)
        return predictions


class MultiQueryTransformerModel(TransformerModel):
    """
    TransformerModel with multi-query decoder.

    Drop-in replacement for ``TransformerModel`` that uses one decoder query token
    per output step instead of a single last-step query.  This gives every future
    step a distinct positional-encoded cross-attention context, eliminating the
    flat-forecast-at-mean collapse seen in the stock Darts implementation.

    All constructor arguments are identical to ``TransformerModel``.
    ``input_chunk_length`` must be >= ``output_chunk_length``.
    """

    def _create_model(self, train_sample: tuple) -> torch.nn.Module:
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _MultiQueryTransformerModule(
            input_size=input_dim,
            output_size=output_dim,
            nr_params=nr_params,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            norm_type=self.norm_type,
            custom_encoder=self.custom_encoder,
            custom_decoder=self.custom_decoder,
            **self.pl_module_params,
        )
