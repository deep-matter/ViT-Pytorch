import numpy as np 
from layer import *
from layerNumpy import *


def TransfomerDecoder(embed_input,mask ,
                      head, Wieghts_QKY , 
                      Wieghts_out ,FullyLinear1, 
                      FullLinear2 , eps,enc=None):
    input_embedding = embed_input.numpy()
    multiHeads , _ = multiHeads_Attention(input_embedding,
                                     Wieghts_QKY.detach().numpy().T,
                                     head,
                                     Wieghts_out.detach().numpy().T,  
                                     mask=None)
    Residual = NormLayar((input_embedding + multiHeads) + eps )
    
    if enc is not None:
        Query , key  = np.split(enc , 2 , axis=-1)
        enc_ = np.concatenate((Query[:,:,:16] , key[:,:,:16]  ,Residual[:,:,:32] ), axis = -1)
        MaskedMUltiHeads ,_= multiHeads_Attention(enc_,
                                         Wieghts_QKY.detach().numpy().T,
                                         head,
                                         Wieghts_out.detach().numpy().T,  
                                         mask=mask.numpy())
        Residual = NormLayar((enc + MaskedMUltiHeads) + eps )
    
    output = NormLayar((Residual + ReLU(np.matmul(Residual,FullyLinear1))@FullLinear2) + eps)
    return output