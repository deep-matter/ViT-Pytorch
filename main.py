from blocks.encoder import * 
from blocks.decoder import * 
import numpy as np

class Transformer:
    def __init__(self, embedding_size , heads):
        super(Transformer,self).__init__()
        
        """
        Blocks models
        -------------
        encoder : is used without the Mask at this stage 
        Decoder : include the Mask and with trick to split the Qurey and Key 
        
        Parametes :
        Wieghst_QKY_Encoder : is Leanrble wieght matrix feed into Encoder
        Wieghst_QKY_Decoder : is Leanrble wieght matrix feed into ncoder
        embed_size : is mebeding_size dimession of input 
        Seq_len : max lenght of vacolublaries 
        Linear_1 : Linear Denes layar of Deooder
        Linear_2 : Linear Denes layar of Deooder after include the ooutput from Encoder
        Linear_encoder : Linear Denes layar of Enooder
        """
    
        self.embed_input = embedding_size
        self.heads = heads 
        self.eps = 1e-12
        self.W_QKY_Encoder = transEncoder.self_attn.in_proj_weight
        self.W_Out_Encoder = transEncoder.self_attn.out_proj.weight
        self.W_QKY_Decoder = transDecoder.self_attn.in_proj_weight
        self.W_Out_Decoder = transDecoder.self_attn.out_proj.weight
        self.Linear1_Encoder = transEncoder.linear1.weight
        self.Linear2_Encoder = transEncoder.linear2.weight
        self.Linear1_Decoder = transDecoder.linear1.weight
        self.Linear2_Decoder = transDecoder.linear2.weight
    
    def forward(self, enc_ , dec_, mask):
        Encoder_= TransfomerEncoder(enc_ ,None ,
                  self.heads, self.W_QKY_Encoder , 
                  self.W_Out_Encoder ,self.Linear1_Encoder.detach().numpy().T, 
                  self.Linear2_Encoder.detach().numpy().T , self.eps)

        Decoder_ = TransfomerDecoder(dec_, mask ,
                  self.heads, self.W_QKY_Decoder , 
                  self.W_Out_Decoder , self.Linear1_Decoder.detach().numpy().T, 
                  self.Linear2_Decoder.detach().numpy().T ,self.eps,enc=Encoder_)

        return Decoder_