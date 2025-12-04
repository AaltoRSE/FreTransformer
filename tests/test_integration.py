import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
from torch.utils.data import DataLoader


class TestIntegration:
    """Integration tests to verify the full pipeline works"""

    def test_data_loading_pipeline(self):
        """Test complete data loading pipeline"""
        from fretransformer.data.data_loader import Dataset_ECG
        
        # Create sample data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data = np.random.randn(200, 5)
            df = pd.DataFrame(data)
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test dataset creation
            train_dataset = Dataset_ECG(
                temp_file, 'train', seq_len=20, pre_len=10,
                type='1', train_ratio=0.7, val_ratio=0.2
            )
            
            # Test dataloader
            dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            
            # Load batch
            batch_idx = 0
            for x, y in dataloader:
                assert x.shape[0] == 4  # batch size
                assert x.shape[1] == 20  # seq_len
                assert x.shape[2] == 5   # features
                
                assert y.shape[0] == 4  # batch size
                assert y.shape[1] == 10  # pre_len
                assert y.shape[2] == 5   # features
                
                batch_idx += 1
                if batch_idx >= 3:
                    break
        
        finally:
            os.unlink(temp_file)

    def test_embedding_to_attention_pipeline(self):
        """Test data flow from embedding through attention"""
        from fretransformer.layers.FEmbed import DataEmbedding
        from fretransformer.layers.FSelfAttention_Family import AttentionLayer, FullAttention
        
        d_model = 32
        n_heads = 4
        
        # Embedding
        embedding = DataEmbedding(c_in=5, d_model=d_model, embed_type='fixed')
        
        # Attention
        attention = AttentionLayer(
            FullAttention(mask_flag=False),
            d_model=d_model,
            n_heads=n_heads
        )
        
        # Forward pass
        x_real = torch.randn(2, 50, 5)
        x_imag = torch.randn(2, 50, 5)
        x = torch.complex(x_real, x_imag)
        embedded = embedding(x, None)

        assert torch.is_complex(embedded)

        output, _ = attention(embedded, embedded, embedded, attn_mask=None)

        assert output.shape == embedded.shape
        assert torch.is_complex(output)

    def test_encoder_decoder_pipeline(self):
        """Test encoder-decoder integration"""
        from fretransformer.layers.FEmbed import DataEmbedding
        from fretransformer.layers.FTransformer_EncDec import (
            Encoder, Decoder, EncoderLayer, DecoderLayer
        )
        from fretransformer.layers.FSelfAttention_Family import AttentionLayer, FullAttention
        
        d_model = 32
        n_heads = 4
        
        # Encoder
        encoder_layers = []
        for _ in range(2):
            attn = AttentionLayer(
                FullAttention(mask_flag=False),
                d_model=d_model,
                n_heads=n_heads
            )
            encoder_layers.append(EncoderLayer(attn, d_model=d_model))
        encoder = Encoder(encoder_layers, norm_layer=torch.nn.LayerNorm(d_model))
        
        # Decoder
        decoder_layers = []
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
            decoder_layers.append(DecoderLayer(self_attn, cross_attn, d_model=d_model))
        decoder = Decoder(decoder_layers, norm_layer=torch.nn.LayerNorm(d_model))
        
        # Pipeline test
        encoder_input = torch.complex(
            torch.randn(2, 100, d_model),
            torch.randn(2, 100, d_model)
        )
        decoder_input = torch.complex(
            torch.randn(2, 50, d_model),
            torch.randn(2, 50, d_model)
        )
        
        enc_out, _ = encoder(encoder_input)
        dec_out = decoder(decoder_input, enc_out)
        
        assert dec_out.shape[1] == 50




if __name__ == '__main__':
    pytest.main([__file__, '-v'])