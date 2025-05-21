import torch
import torch.nn as nn


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        # self.activations = [torch.zeros((self.columns), device=self.dev)]
        self.activations = []
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out, tar):
        """
        tar: batch_size * seq_len, inp corresponding to the position where tar == -100 will be ignored
        """
        # print(inp.shape)
        # assert 1==0
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(tar.shape) == 2: # new
            tar = tar.unsqueeze(0)

        tmp = inp.shape[0]  # bs

        mask = tar.ne(-100) # new
        if isinstance(self.layer, nn.Linear):
            # print("IS:",inp.shape)
            if len(inp.shape) == 3:
                # print("yes")
                inp = inp.reshape((-1, inp.shape[-1]))
            mask = mask.flatten() # new
            mask = mask.to(inp.device) # 
            # print("mask device:", mask.device)
            # print("inp device:", inp.device)
            inp = inp[mask]  # remove -100's # new
            inp = inp.t()
        if self.scaler_row.shape[0] != inp.shape[0]:
            print(f"Resizing scaler_row from {self.scaler_row.shape} to {inp.shape[0]}")
            self.scaler_row = torch.zeros((inp.shape[0]), device=self.dev)
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        # print(inp.shape)

        if True:
            # self.scaler_row = torch.zeros((inp.shape[0]), device=self.dev)
            row_norms = torch.norm(inp, p=2, dim=1) ** 2  # Compute row-wise norms
            if self.scaler_row.shape != row_norms.shape:
                raise ValueError(f"Shape mismatch: scaler_row {self.scaler_row.shape}, row_norms {row_norms.shape}")
            self.scaler_row += row_norms / self.nsamples
        else:
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
        self.activations.append(inp)


# Define WrappedGPT class
class WrappedGPTJMLLM:
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        # self.activations = [torch.zeros((self.columns), device=self.dev)]
        self.activations = []
        self.nsamples = 0
        self.cross_attention_layers = [
            3,
            8,
            13,
            18,
            23,
            28,
            33,
            38 ]
        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out, tar):
        """
        tar: batch_size * seq_len, inp corresponding to the position where tar == -100 will be ignored
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if len(tar.shape) == 2: # new
            tar = tar.unsqueeze(0)


        # print(self.layer_id)
        # print("GPT:", inp.shape)
        # print(tar.shape)
        # assert 1==0
        if self.layer_id in self.cross_attention_layers:
            tmp = inp.shape[0]  # bs
            # mask = tar.ne(-100) # new
            if isinstance(self.layer, nn.Linear):
                # print("IS:",inp.shape)
                if len(inp.shape) == 3:
                    # print("yes")
                    inp = inp.reshape((-1, inp.shape[-1]))
                # mask = mask.flatten() # new
                # inp = inp[mask]  # remove -100's # new
                inp = inp.t()

            self.scaler_row *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp

            inp = inp.type(torch.float32)
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
            self.activations.append(inp)
        else:
            tmp = inp.shape[0]  # bs

            mask = tar.ne(-100) # new
            if isinstance(self.layer, nn.Linear):
                # print("IS:",inp.shape)
                if len(inp.shape) == 3:
                    # print("yes")
                    inp = inp.reshape((-1, inp.shape[-1]))
                mask = mask.flatten() # new
                inp = inp[mask]  # remove -100's # new
                inp = inp.t()

            self.scaler_row *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp

            inp = inp.type(torch.float32)
            self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples
            self.activations.append(inp)
