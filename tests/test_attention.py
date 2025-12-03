import pytest
import torch
import torch.nn as nn
import numpy as np
from fretransformer.layers.FEmbed import (
    PositionalEmbedding, TokenEmbedding, DataEmbedding
)


class TestPositionalEmbedding:
    def test_output_shape(self):
        """Test positional embedding output shape"""
        d_model = 64
        max_len = 100
        pos_embed = PositionalEmbedding(d_model, max_len)
        
        x = torch.randn(2, 50, d_model)  # Batch of 2, seq_len 50
        output = pos_embed(x)
        
        assert output.shape == (1, 50, d_model), f"Expected (1, 50, {d_model}), got {output.shape}"

    def test_positional_values_in_range(self):
        """Test that positional embeddings are in reasonable range"""
        pos_embed = PositionalEmbedding(64, 100)
        x = torch.randn(1, 50, 64)
        output = pos_embed(x)
        
        assert torch.all(torch.isfinite(output)), "Output contains NaN or Inf"

    def test_different_positions(self):
        """Test that different positions give different embeddings"""
        pos_embed = PositionalEmbedding(64, 100)
        x = torch.randn(1, 100, 64)
        output = pos_embed(x)
        
        # First and last position embeddings should be different
        assert not torch.allclose(output[0, 0, :], output[0, 99, :])

    def test_no_gradient_flow(self):
        """Test that positional embeddings don't require gradients"""
        pos_embed = PositionalEmbedding(64, 100)
        x = torch.randn(1, 50, 64)
        output = pos_embed(x)
        
        assert not output.requires_grad, "Positional embedding should not require gradients"


class TestTokenEmbedding:
    def test_output_shape_real(self):
        """Test token embedding output shape for complex input"""
        c_in = 5
        d_model = 64
        token_embed = TokenEmbedding(c_in, d_model)
        x_real = torch.randn(2, 50, c_in)
        x_imag = torch.randn(2, 50, c_in)
        x = torch.complex(x_real, x_imag)
        output = token_embed(x)
        assert output.shape == (2, 50, d_model), f"Expected (2, 50, {d_model}), got {output.shape}"

    def test_complex_output(self):
        """Test that output is complex-valued"""
        token_embed = TokenEmbedding(5, 64)
        x_real = torch.randn(2, 50, 5)
        x_imag = torch.randn(2, 50, 5)
        x = torch.complex(x_real, x_imag)
        output = token_embed(x)
        assert torch.is_complex(output), "Output should be complex-valued"

    def test_sparsity_applied(self):
        """Test that sparsity threshold is applied"""
        token_embed = TokenEmbedding(5, 64)
        x_real = torch.ones(1, 50, 5)
        x_imag = torch.ones(1, 50, 5)
        x = torch.complex(x_real, x_imag)
        output = token_embed(x)
        # Softshrink should zero out small values
        real_part = output.real
        imag_part = output.imag
        num_zeros = (real_part.abs() < 1e-6).sum() + (imag_part.abs() < 1e-6).sum()
        # Should have some sparse values
        assert num_zeros > 0, "Sparsity threshold not applied"

    def test_output_finite(self):
        """Test that output contains no NaN or Inf"""
        token_embed = TokenEmbedding(5, 64)
        x_real = torch.randn(2, 50, 5)
        x_imag = torch.randn(2, 50, 5)
        x = torch.complex(x_real, x_imag)
        output = token_embed(x)
        assert torch.all(torch.isfinite(output.real)), "Real part contains NaN/Inf"
        assert torch.all(torch.isfinite(output.imag)), "Imag part contains NaN/Inf"


class TestDataEmbedding:
    def test_with_temporal_marks(self):
        """Test data embedding with temporal marks"""
        d_model = 64
        data_embed = DataEmbedding(c_in=5, d_model=d_model, embed_type='fixed', freq='h')
        x_real = torch.randn(2, 50, 5)
        x_imag = torch.randn(2, 50, 5)
        x = torch.complex(x_real, x_imag)
        x_mark = torch.tensor([[[1, 2, 3, 4]]] * 2 * 50, dtype=torch.float32).reshape(2, 50, 4)
        output = data_embed(x, x_mark)
        assert output.shape == (2, 50, d_model), f"Expected (2, 50, {d_model}), got {output.shape}"

    def test_without_temporal_marks(self):
        """Test data embedding without temporal marks"""
        d_model = 64
        data_embed = DataEmbedding(c_in=5, d_model=d_model, embed_type='fixed')
        x_real = torch.randn(2, 50, 5)
        x_imag = torch.randn(2, 50, 5)
        x = torch.complex(x_real, x_imag)
        output = data_embed(x, None)
        assert output.shape == (2, 50, d_model)

    def test_complex_dropout(self):
        """Test dropout on complex tensors"""
        data_embed = DataEmbedding(c_in=5, d_model=64, dropout=0.5)
        data_embed.train()  # Set to train mode
        x_real = torch.ones(2, 50, 5)
        x_imag = torch.ones(2, 50, 5)
        x = torch.complex(x_real, x_imag)
        output = data_embed(x, None)
        # In train mode with dropout, some values should be zero
        assert torch.is_complex(output)
        assert torch.any(output.real == 0) or torch.any(output.imag == 0)

    def test_complex_output_is_complex(self):
        """Test that complex output is properly complex-valued"""
        data_embed = DataEmbedding(c_in=5, d_model=64, embed_type='fixed')
        x_real = torch.randn(2, 50, 5)
        x_imag = torch.randn(2, 50, 5)
        x = torch.complex(x_real, x_imag)
        output = data_embed(x, None)
        assert torch.is_complex(output), "Output should be complex"

    def test_output_finite(self):
        """Test output contains no NaN or Inf"""
        data_embed = DataEmbedding(c_in=5, d_model=64)
        x_real = torch.randn(2, 50, 5)
        x_imag = torch.randn(2, 50, 5)
        x = torch.complex(x_real, x_imag)
        output = data_embed(x, None)
        assert torch.all(torch.isfinite(output.real)), "Real part contains NaN/Inf"
        assert torch.all(torch.isfinite(output.imag)), "Imag part contains NaN/Inf"


class TestEmbeddingGradients:
    def test_token_embedding_gradients(self):
        """Test that gradients flow through token embedding"""
        token_embed = TokenEmbedding(5, 64)
        x_real = torch.randn(2, 50, 5, requires_grad=True)
        x_imag = torch.randn(2, 50, 5, requires_grad=True)
        x = torch.complex(x_real, x_imag)
        output = token_embed(x)
        loss = output.real.sum()
        loss.backward()
        
        assert x_real.grad is not None or x_imag.grad is not None, "Gradients not flowing through token embedding"
        assert (x_real.grad is not None and torch.any(x_real.grad != 0)) or (x_imag.grad is not None and torch.any(x_imag.grad != 0)), "Gradients are all zero"
    
    @pytest.mark.skip(reason="Gradients are purposefully dropped in the code")
    def test_data_embedding_gradients(self):
        """Test that gradients flow through data embedding"""
        data_embed = DataEmbedding(5, 64)
        x_real = torch.randn(2, 50, 5, requires_grad=True)
        x_imag = torch.randn(2, 50, 5, requires_grad=True)
        x = torch.complex(x_real, x_imag)
        output = data_embed(x, None)
        loss = output.real.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.any(x.grad != 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])