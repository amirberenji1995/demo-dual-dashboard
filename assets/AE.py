import torch

class AE(torch.nn.Module):
  def __init__(self):
    super(AE, self).__init__()
    self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 256),
            torch.nn.Tanh(),
        )
    self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 2048),
            torch.nn.Tanh(),
        )
  def forward(self, x):
          encoded = self.encoder(x)
          decoded = self.decoder(encoded)
          return decoded