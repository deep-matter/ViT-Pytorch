import numpy as np 
from layer import * 
from layerNumpy import *

def TransfomerEncoder(embed_input,mask ,
                      head, Wieghts_QKY , 
                      Wieghts_out ,FullyLinear1, 
                      FullLinear2 , eps):
    input_embedding = embed_input.numpy()
    multiHeads , _ = multiHeads_Attention(input_embedding,
                                     Wieghts_QKY.detach().numpy().T,
                                     head,
                                     Wieghts_out.detach().numpy().T,  
                                     mask=None)
    Residual = NormLayar((input_embedding + multiHeads) + eps )
    
    output = NormLayar((Residual + ReLU(np.matmul(Residual,FullyLinear1))@FullLinear2) + eps)
    return output