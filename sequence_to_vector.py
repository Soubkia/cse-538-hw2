'''
author: Sounak Mondal
'''

# std lib imports
from typing import Dict

# external libs
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

class SequenceToVector(nn.Module):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``torch.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``torch.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : torch.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : torch.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2, device = 'cpu'):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.dropout = dropout
        # Create the layers with ReLU activation.
        layers = [
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        ] * (num_layers - 1)
        layers.append(nn.Linear(input_dim, input_dim))
        self.layers = nn.Sequential(*layers)
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        # TODO: Store the layer representations.
        masked_sequence = vector_sequence * sequence_mask.unsqueeze(2)
        if training:
            r_w = torch.distributions.bernoulli.Bernoulli(self.dropout)
            dropout_mask = r_w.sample(vector_sequence.shape[:2])
            masked_sequence *= dropout_mask.unsqueeze(2)
        avg = masked_sequence.mean(dim=1)
        combined_vector = self.layers(avg)
        layer_representations = []
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int, device = 'cpu'):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.layers = nn.GRU(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # TODO(students): end

    def forward(self,
             vector_sequence: torch.Tensor,
             sequence_mask: torch.Tensor,
             training=False) -> torch.Tensor:
        # TODO(students): start
        # TODO: Properly store layer representations.
        masked_sequence = vector_sequence * sequence_mask.unsqueeze(2)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            masked_sequence,
            sequence_mask.sum(axis=1),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, layer_representations = self.layers(packed_sequence)
        combined_vector = layer_representations[-1]
        #  unpacked_output, lengths = nn.utils.rnn.pad_packed_sequence(
        #      packed_output, batch_first=True
        #  )
        #  combined_vector = unpacked_output[:, 0]
        #  combined_vector = unpacked_output[:, lengths - 1][:, 0]
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
