import torch
from torch import nn
from torch.nn import functional as F


class Highway(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,
        activation=F.relu,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList(nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers))
        self.activation = activation
        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * current_input + (1 - gate) * nonlinear_part
        return current_input


class CharacterCNN(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int = 768,
        character_embedding_dim: int = 16,
        n_characters: int = 262,
        max_characters_per_token: int = 50,
        filters: list[tuple[int, int]] | None = None,
        n_highway: int = 2,
        activation: str = "relu",
        requires_grad: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.character_embedding_dim = character_embedding_dim
        self.n_characters = n_characters
        self.max_characters_per_token = max_characters_per_token
        self.filters = filters or [
            (1, 32),
            (2, 32),
            (3, 64),
            (4, 128),
            (5, 256),
            (6, 512),
            (7, 1024),
        ]
        self.activation_name = activation

        self._char_embedding_weights = nn.Parameter(
            torch.zeros(n_characters + 1, character_embedding_dim),
            requires_grad=requires_grad,
        )

        convolutions = []
        for index, (width, num_filters) in enumerate(self.filters):
            convolution = nn.Conv1d(character_embedding_dim, num_filters, kernel_size=width)
            self.add_module(f"char_conv_{index}", convolution)
            convolutions.append(convolution)
        self._convolutions = convolutions
        n_filters = sum(num_filters for _, num_filters in self.filters)
        self._highways = Highway(n_filters, n_highway, activation=F.relu)
        self._projection = nn.Linear(n_filters, output_dim)

        for parameter in self.parameters():
            parameter.requires_grad = requires_grad

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        character_embedding = F.embedding(
            inputs.view(-1, self.max_characters_per_token),
            self._char_embedding_weights,
        )
        character_embedding = character_embedding.transpose(1, 2)

        if self.activation_name == "tanh":
            activation = torch.tanh
        elif self.activation_name == "relu":
            activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

        conv_outputs = []
        for convolution in self._convolutions:
            convolved = convolution(character_embedding)
            convolved, _ = torch.max(convolved, dim=-1)
            conv_outputs.append(activation(convolved))

        token_embedding = torch.cat(conv_outputs, dim=-1)
        token_embedding = self._highways(token_embedding)
        token_embedding = self._projection(token_embedding)

        batch_size, sequence_length, _ = inputs.size()
        return token_embedding.view(batch_size, sequence_length, -1)
