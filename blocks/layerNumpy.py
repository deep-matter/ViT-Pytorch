import numpy as np 


def NormLayar(Z):
    mean = Z.mean(axis=-1 , keepdims=True)
    var = Z.var(axis=-1 , keepdims = True)
    return ((Z - mean) / np.sqrt(var)) + eps

def ReLU(Z):
    return np.maximum(Z,0)
    

def Softmax(z):
    """
    descriptions function : Softmax is non-linear function that give the averege of between 
    0 and 1 of in element in matrix 
    """
    e_x = np.exp(z - z.max(axis=-1,keepdims=True))
    return e_x / np.sum(e_x , axis=-1 ,keepdims=True)

def Self_Attention(input_embedding ,WieghtMatrix_QKY, out_wieghts,mask=None,batch_first=True) :
    """
    Self-Attention take input of emebeding matrix which asseccoite with 
    the Positional encoding ww will cover later in section 
    Query and Key and Value all of them have the same dimession as the input 
    """
    try : 
        if batch_first==True:
            Query , Key , value = np.split(input_embedding@WieghtMatrix_QKY , 3 , axis=-1)
            if mask is not None:
                assert mask.shape[0] == input_embedding.shape[1],\
                    f"input dimession of mask doesn't match with dimession of embedding input:{mask.shape[0]} {input_embedding.shape[0]}"
                Attention = Softmax(Key@Query.swapaxes(-1,-2) / np.sqrt(input_embedding.shape[-1]) + mask) 
                return  Attention@value@out_wieghts , Attention
            else:
                Attention = Softmax(Key@Query.swapaxes(-1,-2) / np.sqrt(input_embedding.shape[-1])) 
                return  Attention@value@out_wieghts , Attention
    except:
        raise Exception("Batch argumment is missing")
