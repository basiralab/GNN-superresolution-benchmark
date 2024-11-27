import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        B, N, _ = input.shape
        h = torch.matmul(input, self.W)  # Shape: [B, N, out_features]

        # Prepare the repeat and repeat_interleave tensors for a batch
        f_repeat = h.repeat(1, 1, N).view(B, N * N, -1)
        f_repeat_interleave = h.repeat_interleave(N, dim=1)

        # Concatenate
        a_input = torch.cat([f_repeat, f_repeat_interleave], dim=2)  # Shape: [B, N*N, 2*out_features]

        # Compute e using a single forward pass for all pairs
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        e = e.view(B, N, N)  # Reshape e to [B, N, N] to match adj

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # Apply softmax over the second dimension (N)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)  # New shape: [B, N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adjacency):
        temp = torch.matmul(input, self.weight)
        output = torch.matmul(adjacency, temp)
        return output
    

class Generator_(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator_, self).__init__()
        self.gat1 = GATLayer(input_size, 4)
        self.gcn1 = GCNLayer(4, 8)
        self.flatten = nn.Flatten(start_dim=1)
        self.l1 = nn.Linear(input_size*8, input_size*16)
        self.l2 = nn.Linear(input_size*16, output_size)

    def forward(self, adj, noise):
        x = F.relu(self.gat1(noise, adj))
        x = F.relu(self.gcn1(x, adj))
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.tanh(self.l2(x))
        return x
    

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, output_size),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    # Simplified discriminator model
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class GraphCycleGAN(nn.Module):
    def __init__(self, input_A_size, input_B_size):
        super(GraphCycleGAN, self).__init__()
        # Initialize generator and discriminator for both domains
        self.G_A2B = Generator(input_A_size, input_B_size)
        self.G_B2A = Generator(input_B_size, input_A_size)
        self.D_A = Discriminator(input_A_size)
        self.D_B = Discriminator(input_B_size)

    def forward(self, x):
        # This method might not be used directly for CycleGANs
        pass