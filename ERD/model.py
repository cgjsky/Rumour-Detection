from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class pooling_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(pooling_layer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, inputs):
        inputs=inputs.permute(0,2,1)
        inputs=self.linear(inputs)
        inputs_sent,_=torch.max(inputs,2,keepdim=False)
        #inputs_sent = [torch.cat([self.linear(sent_tensor).max(axis=0)[0].unsqueeze(0) for sent_tensor in seq]) for seq in inputs]
        
        return inputs_sent

class RDM_Layer(nn.Module):
    def __init__(self, word_embedding_dim, sent_embedding_dim, hidden_dim, dropout_prob):
        super(RDM_Layer, self).__init__()
        self.embedding_dim = sent_embedding_dim
        self.hidden_dim = hidden_dim
        self.gru_model = nn.GRU(word_embedding_dim, 
                                self.hidden_dim, 
                                batch_first=True, 
                                dropout=dropout_prob
                            )
        self.DropLayer = nn.Dropout(dropout_prob)

    def forward(self, input_x): 
        """
        input_x: [batchsize, max_seq_len, sentence_embedding_dim] 
        x_len: [batchsize]
        init_states: [batchsize, hidden_dim]
        """
        batchsize, max_seq_len, emb_dim = input_x.shape
        init_states = torch.zeros([1, batchsize, self.hidden_dim], dtype=torch.float32)
        df_outputs, df_last_state = self.gru_model(input_x, init_states)
        return df_outputs
        
class RDM_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_layer=pooling_layer(50,50)
        self.RDM_layer=RDM_Layer(6,50,256,0.2)
        self.fc=nn.Linear(256,2)
    
    def forward(self,x):
        #x->[batch_size,word_length,embedding_dim]
        out = self.pool_layer(x)
        out = out.reshape(16,50,-1)
        out = self.RDM_layer(out)
        out = self.fc(out)
        #out->[batch_size,max_seq_len,2]
        return out

rdm_model=RDM_Model()
criteration=nn.CrossEntropyLoss()

optim=torch.optim.Adam(rdm_model.parameters(),lr=5e-3)

class CM_Model(nn.Module):
    def __init__(self, hidden_dim, action_num):
        super(CM_Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.action_num = action_num
        self.DenseLayer = nn.Linear(self.hidden_dim, 64)
        self.Classifier = nn.Linear(64, self.action_num)
        
    def forward(self, rdm_state):
        """
        rdm_state: [batchsize, hidden_dim]
        """
        batchsize, hidden_dim = rdm_state.shape
        rl_h1 = nn.functional.relu(
            self.DenseLayer(
                rdm_state
            )
        )
        stopScore = self.Classifier(rl_h1)
        isStop = stopScore.argmax(axis=1)
        return stopScore, isStop
