
import torch
from torch import nn
class SCLazyLinear(nn.Module):
    def __init__(self, args):
        super(SCLazyLinear, self).__init__()
        self.activation = self.get_active_func(args)
        self.args = args
        self.device=args.device
        self.conv = nn.Sequential(
            nn.Conv2d(10,20,3),
            self.activation,
            nn.Conv2d(20,40,(3)),
            self.activation,
            nn.Conv2d(40,80,3,padding=1),
            self.activation
        ).to(args.device)
        self.rnn = nn.GRU(
            input_size=240,
            num_layers=args.GRU_n_layers,
            hidden_size=args.rnn_hidden_dim,
            batch_first=False
        ).to(self.device)
        self.rnn.flatten_parameters()
        #DSO REN YOU END

        self.final_layer = nn.LazyLinear( args.n_actions,device=self.device)
        #DSO Nicholas END

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    #DSO Nicholas
    def get_active_func(self, args):
        
        # set activation function for model optimisation
        if args.activation == "relu":
            return nn.ReLU()
        elif args.activation == "tanh":
            return nn.Tanh()
        elif args.activation == "sigmoid":
            return nn.Sigmoid()
        elif args.activation == "leaky_relu":
            return nn.LeakyReLU()
        elif args.activation == "selu":
            return nn.SELU()
        elif args.activation == "silu":
            return nn.SiLU()
        elif args.activation == "hardswish":
            return nn.Hardswish()
        elif args.activation == "identity":
            return nn.Identity()
        elif args.activation == "prelu":
            return nn.PReLU()
        else:
            assert False, "activation function not supported!"
    #DSO Nicholas END

    def forward(self, inputs, hidden_state=None):
        if hidden_state == None:
            raise Exception("Needs hidden state, will not automatically assume it exists")
        curr = self.conv(inputs.to(self.device))
        rnn_input = torch.flatten(curr).unsqueeze(0)
        print(rnn_input.shape)
        # k+=1
        curr = self.rnn(rnn_input,hidden_state)
        Q_value = self.final_layer(curr[0])
        return Q_value,curr[1]
if __name__ == "__main__":
    from arguments import get_common_args
    args = get_common_args()
    print(args)
    temp = SCLazyLinear(args)
    
    temp2 = temp(torch.zeros((1,10,5,7)),torch.zeros(3,64).to(args.device))
    print(temp2[0].shape)
    print(temp2[1].shape)