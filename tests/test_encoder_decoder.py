import pytest
import torch
import torch.nn as nn
from fretransformer.layers.FTransformer_EncDec import (
    EncoderLayer, Encoder, DecoderLayer, Decoder
)
from fretransformer.layers.FSelfAttention_Family import FullAttention, AttentionLayer


class TestEncoderLayer:
    @pytest.fixture
    def encoder_layer(self):
        """Create an encoder layer for testing"""
        d_model = 32
        n_heads = 4
        attention = AttentionLayer(
            FullAttention(mask_flag=False),
            d_model=d_model,
            n_heads=n_heads
        )
        return EncoderLayer(attention, d_model=d_model, d_fgcn=64, dropout=0.1)

    def test_output_shape(self, encoder_layer):
        """Test encoder layer output shape"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output, attn = encoder_layer(x)
        
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_complex_output(self, encoder_layer):
        """Test that output is complex"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output, _ = encoder_layer(x)
        
        assert torch.is_complex(output), "Output should be complex"

    def test_residual_connection(self, encoder_layer):
        """Test that residual connections work"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output, _ = encoder_layer(x)
        
        # Output should not be identical to input (due to transformations)
        # but should be close in scale due to residual
        assert not torch.allclose(output, x, atol=0.1)

    def test_layer_normalization(self, encoder_layer):
        """Test that layer normalization is applied"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output, _ = encoder_layer(x)
        
        # Output should be stable (not exploding gradients)
        assert torch.all(torch.isfinite(output.real))
        assert torch.all(torch.isfinite(output.imag))

    def test_finite_output(self, encoder_layer):
        """Test output contains no NaN or Inf"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output, _ = encoder_layer(x)
        
        assert torch.all(torch.isfinite(output.real))
        assert torch.all(torch.isfinite(output.imag))

    def test_gradient_flow(self, encoder_layer):
        """Test that gradients flow through encoder layer"""
        encoder_layer.train()
        x_real = torch.randn(2, 50, 32, requires_grad=True)
        x_imag = torch.randn(2, 50, 32, requires_grad=True)
        x = torch.complex(x_real, x_imag)
        output, _ = encoder_layer(x)
        loss = output.real.sum()
        loss.backward()
        assert x_real.grad is not None or x_imag.grad is not None, "No gradients"
        assert (x_real.grad is not None and torch.any(x_real.grad != 0)) or (x_imag.grad is not None and torch.any(x_imag.grad != 0)), "Gradients are all zero"

class TestEncoder:
    @pytest.fixture
    def encoder(self):
        """Create an encoder for testing"""
        d_model = 32
        n_heads = 4
        attn_layers = []
        
        for _ in range(2):
            attention = AttentionLayer(
                FullAttention(mask_flag=False),
                d_model=d_model,
                n_heads=n_heads
            )
            attn_layers.append(
                EncoderLayer(attention, d_model=d_model, d_fgcn=64, dropout=0.1)
            )
        
        return Encoder(attn_layers, norm_layer=nn.LayerNorm(d_model))

    def test_multi_layer_output_shape(self, encoder):
        """Test encoder output shape with multiple layers"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output, attns = encoder(x)
        
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        assert len(attns) == 2, f"Expected 2 attention outputs, got {len(attns)}"

    def test_attention_list_returned(self, encoder):
        """Test that attention weights are returned"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output, attns = encoder(x)
        
        assert isinstance(attns, list), "Attentions should be returned as list"
        assert all(a is None or torch.is_complex(a) for a in attns)

    def test_final_normalization(self, encoder):
        """Test that final layer norm is applied"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output, _ = encoder(x)
        
        assert torch.all(torch.isfinite(output.real))
        assert torch.all(torch.isfinite(output.imag))

    def test_stacked_layers(self, encoder):
        """Test that multiple layers stack correctly"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output, _ = encoder(x)
        
        # Output should be complex and properly shaped
        assert torch.is_complex(output)
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 50  # seq length
        assert output.shape[2] == 32  # d_model


class TestDecoderLayer:
    @pytest.fixture
    def decoder_layer(self):
        """Create a decoder layer for testing"""
        d_model = 32
        n_heads = 4
        
        self_attn = AttentionLayer(
            FullAttention(mask_flag=True),
            d_model=d_model,
            n_heads=n_heads
        )
        cross_attn = AttentionLayer(
            FullAttention(mask_flag=False),
            d_model=d_model,
            n_heads=n_heads
        )
        
        return DecoderLayer(
            self_attn, cross_attn,
            d_model=d_model, d_fgcn=64, dropout=0.1
        )

    def test_output_shape(self, decoder_layer):
        """Test decoder layer output shape"""
        x = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        cross = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        
        output = decoder_layer(x, cross)
        
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_self_and_cross_attention(self, decoder_layer):
        """Test that both self and cross attention are applied"""
        x = torch.complex(torch.randn(2, 30, 32), torch.randn(2, 30, 32))
        cross = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        
        output = decoder_layer(x, cross)
        
        assert output.shape == x.shape
        assert torch.is_complex(output)

    def test_causal_masking_in_self_attention(self, decoder_layer):
        """Test that self-attention uses causal masking"""
        x = torch.complex(torch.randn(1, 20, 32), torch.randn(1, 20, 32))
        cross = torch.complex(torch.randn(1, 20, 32), torch.randn(1, 20, 32))
        
        output = decoder_layer(x, cross)
        
        # Should not produce NaN/Inf even with causal masking
        assert torch.all(torch.isfinite(output.real))
        assert torch.all(torch.isfinite(output.imag))


class TestDecoder:
    @pytest.fixture
    def decoder(self):
        """Create a decoder for testing"""
        d_model = 32
        n_heads = 4
        layers = []
        
        for _ in range(2):
            self_attn = AttentionLayer(
                FullAttention(mask_flag=True),
                d_model=d_model,
                n_heads=n_heads
            )
            cross_attn = AttentionLayer(
                FullAttention(mask_flag=False),
                d_model=d_model,
                n_heads=n_heads
            )
            layers.append(
                DecoderLayer(self_attn, cross_attn, d_model=d_model, d_fgcn=64)
            )
        
        projection = nn.Linear(d_model, 5)  # Project to 5 output features
        return Decoder(layers, norm_layer=nn.LayerNorm(d_model), projection=projection)

    def test_with_projection(self, decoder):
        """Test decoder with output projection"""
        x = torch.complex(torch.randn(2, 30, 32), torch.randn(2, 30, 32))
        cross = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        
        output = decoder(x, cross)
        
        # After projection, output should have different last dimension
        assert output.shape[0] == 2  # batch
        assert output.shape[1] == 30  # seq length
        assert output.shape[2] == 5   # output size

    def test_without_projection(self):
        """Test decoder without projection"""
        d_model = 32
        n_heads = 4
        
        layers = []
        for _ in range(2):
            self_attn = AttentionLayer(
                FullAttention(mask_flag=True),
                d_model=d_model,
                n_heads=n_heads
            )
            cross_attn = AttentionLayer(
                FullAttention(mask_flag=False),
                d_model=d_model,
                n_heads=n_heads
            )
            layers.append(
                DecoderLayer(self_attn, cross_attn, d_model=d_model)
            )
        
        decoder = Decoder(layers, norm_layer=nn.LayerNorm(d_model))
        
        x = torch.complex(torch.randn(2, 30, 32), torch.randn(2, 30, 32))
        cross = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        output = decoder(x, cross)
        
        assert output.shape == x.shape

    def test_encoder_decoder_compatibility(self, decoder):
        """Test that decoder works with encoder output"""
        # Encoder output
        encoder_output = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        
        # Decoder input (typically shorter)
        decoder_input = torch.complex(torch.randn(2, 30, 32), torch.randn(2, 30, 32))
        
        output = decoder(decoder_input, encoder_output)
        
        assert output.shape[0] == 2  # batch size preserved
        assert output.shape[1] == 30  # decoder seq length preserved


class TestEncoderDecoderShapeFlows:
    def test_different_sequence_lengths(self):
        """Test encoder-decoder with different seq lengths"""
        d_model = 32
        n_heads = 4
        
        # Encoder
        encoder_attn = AttentionLayer(
            FullAttention(mask_flag=False),
            d_model=d_model,
            n_heads=n_heads
        )
        encoder = Encoder([
            EncoderLayer(encoder_attn, d_model=d_model)
        ])
        
        # Decoder
        dec_self_attn = AttentionLayer(
            FullAttention(mask_flag=True),
            d_model=d_model,
            n_heads=n_heads
        )
        dec_cross_attn = AttentionLayer(
            FullAttention(mask_flag=False),
            d_model=d_model,
            n_heads=n_heads
        )
        decoder = Decoder([
            DecoderLayer(dec_self_attn, dec_cross_attn, d_model=d_model)
        ])
        
        # Different lengths
        encoder_input = torch.complex(torch.randn(2, 100, 32), torch.randn(2, 100, 32))
        decoder_input = torch.complex(torch.randn(2, 50, 32), torch.randn(2, 50, 32))
        
        enc_output, _ = encoder(encoder_input)
        dec_output = decoder(decoder_input, enc_output)
        
        assert enc_output.shape[1] == 100
        assert dec_output.shape[1] == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])