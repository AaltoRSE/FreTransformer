from torchview import draw_graph
from fretransformer.model.Fourier_Transformer import FTransformer
import torch

if __name__ == "__main__":
    # Example config, adjust as needed
    class Args:
        embed_size = 128
        hidden_size = 256
        pre_length = 20
        enc_in = 5
        seq_length = 20
        task_name = 'long_term_forecast'
        d_model = 16
        n_heads = 8
        d_fgcn = 32
        activation = 'gelu'
        e_layers = 2
        dec_in = 5
        factor = 1
        c_out = 5
        d_layers = 1
        label_len = 48
        dropout = 0.0
        output_attention = False
        embed = 'timeF'
        freq = 'h'

    args = Args()
    model = FTransformer(args)
    batch_size = 2
    seq_length = args.seq_length
    enc_in = args.enc_in
    dec_in = args.dec_in
    x = torch.zeros((batch_size, seq_length, enc_in))
    dec_inp = torch.zeros((batch_size, seq_length, dec_in))
    model_graph = draw_graph(model, input_data=(x, dec_inp), device='meta')
    # Save as PNG
    model_graph.visual_graph.render("ftransformer_graph", format="png")