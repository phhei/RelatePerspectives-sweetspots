from abc import ABC, abstractmethod
from typing import Union, Tuple, Iterable, List, Optional
from typing_extensions import Literal
from functools import reduce

import torch
import operator


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class UserTowerInterface(torch.nn.Module, ABC):
    @abstractmethod
    def get_embedding_length(self) -> int:
        pass


class OneHotEncoder(UserTowerInterface):
    def __init__(self, all_user_ids: List[int], one_hot_vector_padding: Optional[int] = None):
        super().__init__()

        self.mapping = {_id: i for i, _id in enumerate(all_user_ids)}
        self.padding = one_hot_vector_padding or 0

    def forward(self, user_ids: torch.LongTensor) -> torch.Tensor:
        one_hot_encoding = torch.nn.functional.one_hot(
            torch.tensor(data=[self.mapping.get(u.item(), len(self.mapping)+self.padding-1) for u in user_ids.cpu()],
                         dtype=torch.long),
            num_classes=len(self.mapping)+self.padding
        ).to(torch.float)

        if torch.cuda.is_available():
            return one_hot_encoding.cuda()

        return one_hot_encoding

    def get_embedding_length(self) -> int:
        return len(self.mapping)+self.padding


class LinearEncoder(OneHotEncoder):
    def __init__(self, all_user_ids: List[int], embedding_size: int, num_layers: int = 3,
                 activation_function: Optional[torch.nn.Module] = None, dropout: Optional[float] = None):
        super(LinearEncoder, self).__init__(all_user_ids=all_user_ids, one_hot_vector_padding=None)

        self.embedding_size = embedding_size
        self.neural_net = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=len(self.mapping) if i == 0 else embedding_size, out_features=embedding_size, bias=i > 0
                ),
                torch.nn.Identity() if dropout is None else torch.nn.Dropout(p=dropout),
                torch.nn.Identity() if activation_function is None else activation_function
            ) for i in range(num_layers)]
        )

        if torch.cuda.is_available():
            self.neural_net.cuda()

    def forward(self, user_ids: torch.LongTensor) -> torch.Tensor:
        return self.neural_net(super(LinearEncoder, self).forward(user_ids=user_ids))

    def get_embedding_length(self) -> int:
        return self.embedding_size


class CombinerInterface(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(self, user_embedding_length: int, output_shape: Union[int, Tuple[int]]):
        """
        Instantiate a combiner (combining the text and user tower)

        :param user_embedding_length: the embedding length outputted by the user tower (last dimension). In many
        implemented combiners, you can specialize the text_embedding_length, too (outputted by the text tower) - this
        is not necessary by using the standard argperspectives.models.Recommender.Recommender since this implementation
        maps the text embeddings to the user embedding length, hence text_embedding_length == user_embedding_length
        :param output_shape: the desired output shape. Can be a simple number (of classes) or a nestled output class
        (useful for multi-task-settings) structure by using a tuple here
        """
        super().__init__()
        self.user_embedding_length = user_embedding_length
        self.output_shape = output_shape

    @abstractmethod
    def forward(
            self,
            text_embeddings: Union[torch.Tensor, Iterable[torch.Tensor]],
            user_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        :param text_embeddings: one or more tensors (in case of multiple transformers)
        of shape (#texts, #embedding_length)
        :param user_embeddings: a tensor of shape (#users, #embedding_length)
        :return: a tensor of shape (#texts, #users, output_shape)
        """
        if isinstance(text_embeddings, torch.Tensor):
            text_embeddings = (text_embeddings, )
        assert all(map(lambda i: i.shape[-1] in self.get_embedding_in_lengths(), (*text_embeddings, user_embeddings))), \
            f"Your inputs have an unexpected shape: {' and '.join(map(lambda i: str(i.shape), (*text_embeddings, user_embeddings)))}"

    def get_embedding_in_lengths(self) -> Tuple[Optional[int], int]:
        """
        Get the expected IN-embedding lengths

        :return: text_embedding_length (or None if ignored/ not important to know), user_embedding_length
        """
        return None, self.user_embedding_length

    def get_out_classes(self) -> int:
        """
        Get the total number of outputted classes (in case of multidimensional output-shape the product)

        :return: the total number of output classes which are calculated
        """
        return self.output_shape if isinstance(self.output_shape, int) else prod(self.output_shape)


class MatMulCombiner(CombinerInterface):
    def __init__(self, user_embedding_length: int, output_shape: Union[int, Tuple[int]]):
        super(MatMulCombiner, self).__init__(user_embedding_length=user_embedding_length, output_shape=output_shape)

        self.classification_head = torch.nn.Linear(
            in_features=user_embedding_length,
            out_features=output_shape if isinstance(output_shape, int) else prod(output_shape),
            bias=False,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def forward(self, text_embeddings: Union[torch.Tensor, Iterable[torch.Tensor]],
                user_embeddings: torch.Tensor) -> torch.Tensor:
        super(MatMulCombiner, self).forward(text_embeddings=text_embeddings, user_embeddings=user_embeddings)

        if not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = torch.mean(input=torch.stack(tensors=text_embeddings, dim=0), dim=0, keepdim=False)

        expanded_text_embeddings = \
            text_embeddings.repeat((user_embeddings.shape[0], 1)).reshape(
                (text_embeddings.shape[0], user_embeddings.shape[0], self.user_embedding_length)
            )
        expanded_user_embeddings = \
            user_embeddings.repeat((text_embeddings.shape[0], 1)).reshape(
                (text_embeddings.shape[0], user_embeddings.shape[0], self.user_embedding_length)
            )
        
        matmul_lin = self.classification_head(expanded_text_embeddings * expanded_user_embeddings)
        return matmul_lin if isinstance(self.output_shape, int) else matmul_lin.reshape(
            (text_embeddings.shape[0], user_embeddings.shape[0], *self.output_shape)
        )


def min_max_combine(
        text_embeddings: Union[torch.Tensor, Iterable[torch.Tensor]],
        user_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Combines a (set of) text-embedding-batches and user-embeddings to a single tensor

    :param text_embeddings: the (set of) batched text embeddings. Each text embedding tensor must be in the same shape
    as follows: (#texts, text_embedding_length).
    If you insert here multiple (list of) batched text embedding tensors,
    the minimum and the maximum values are considered
    If you insert a single batched text embedding tensor, the tensor is repeated twice.
    :param user_embeddings: the batched user embeddings in the shape of (#users, user_embedding_length)
    :return: a single tensor if shape (#texts, #users, 2*text_embedding_length+user_embedding_length)
    """
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings_min = text_embeddings
        text_embeddings_max = text_embeddings
    else:
        text_embeddings_min, _ = \
            torch.min(input=torch.stack(tensors=text_embeddings, dim=0), dim=0, keepdim=False)
        text_embeddings_max, _ = \
            torch.max(input=torch.stack(tensors=text_embeddings, dim=0), dim=0, keepdim=False)

    return torch.stack(
        tensors=[torch.stack(
            tensors=[torch.concat(
                tensors=(text_embedding_min, text_embedding_max, user_embedding),
                dim=-1
            ) for user_embedding in user_embeddings],
            dim=0
        ) for text_embedding_min, text_embedding_max in zip(text_embeddings_min, text_embeddings_max)],
        dim=0
    )


class LinearCombiner(CombinerInterface):
    def __init__(
            self,
            user_embedding_length: int,
            output_shape: Union[int, Tuple[int]], num_layers: int = 3,
            activation_function: Optional[torch.nn.Module] = None,
            dropout: Optional[float] = None,
            text_embedding_length: Optional[int] = None,
    ):
        super(LinearCombiner, self).__init__(user_embedding_length=user_embedding_length, output_shape=output_shape)

        self.text_embedding_length = text_embedding_length or self.user_embedding_length

        output_shape_lin = output_shape if isinstance(output_shape, int) else prod(output_shape)
        self.neural_net = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.text_embedding_length*2+self.user_embedding_length,
                    out_features=output_shape_lin if i == (num_layers -1) else
                    self.text_embedding_length*2+self.user_embedding_length,
                    bias=True
                ),
                torch.nn.Identity() if dropout is None else torch.nn.Dropout(p=dropout),
                torch.nn.Identity() if activation_function is None else activation_function
            ) for i in range(num_layers)]
        )

        if torch.cuda.is_available():
            self.neural_net.cuda()

    def forward(self, text_embeddings: Union[torch.Tensor, Iterable[torch.Tensor]],
                user_embeddings: torch.Tensor) -> torch.Tensor:
        super(LinearCombiner, self).forward(text_embeddings=text_embeddings, user_embeddings=user_embeddings)

        ret = self.neural_net(min_max_combine(text_embeddings=text_embeddings, user_embeddings=user_embeddings))
        return ret if isinstance(self.output_shape, int) else ret.reshape(
            (text_embeddings.shape[0], user_embeddings.shape[0], *self.output_shape)
        )

    def get_embedding_in_lengths(self) -> Tuple[Optional[int], int]:
        return self.text_embedding_length, self.user_embedding_length


class DeepCrossNetwork(CombinerInterface):
    """
    A combiner combining the tower-processed text-embedding and the tower-processed user-id-embedding
    in the neural way of

    DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems
    by Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi
    """

    class CrossModule(torch.nn.Module):
        def __init__(self, features: int):
            super().__init__()

            self.linear_core = torch.nn.Linear(
                in_features=features,
                out_features=features,
                bias=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

        def forward(self, x_in: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
            return (x_0 * self.linear_core(x_in)) + x_in

    def __init__(
            self,
            user_embedding_length: int,
            output_shape: Union[int, Tuple[int]],
            form: Literal["stacked", "parallel"] = "stacked",
            num_layers: int = 3,
            activation_function: Optional[torch.nn.Module] = None,
            dense_feature_appendix: Optional[int] = None,
            text_embedding_length: Optional[None] = None
    ):
        super(DeepCrossNetwork, self).__init__(user_embedding_length=user_embedding_length, output_shape=output_shape)

        self.form = form
        self.text_embedding_length = text_embedding_length or self.user_embedding_length

        internal_embedding_length = self.text_embedding_length * 2 + self.user_embedding_length
        if dense_feature_appendix is None:
            self.in_embedding_appendix_module = None
        else:
            self.in_embedding_appendix_module = torch.nn.Linear(
                in_features=internal_embedding_length,
                out_features=dense_feature_appendix,
                bias=False,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            internal_embedding_length += dense_feature_appendix

        self.cross_layers = torch.nn.ModuleList(
            [DeepCrossNetwork.CrossModule(features=internal_embedding_length) for _ in
             range(num_layers)]
        )
        self.linear_layers = torch.nn.Sequential(
            *[torch.nn.Sequential(
                torch.nn.Linear(in_features=internal_embedding_length,
                                out_features=internal_embedding_length,
                                bias=True),
                torch.nn.Identity() if activation_function is None else activation_function
            ) for _ in range(num_layers)]
        )

        if torch.cuda.is_available():
            self.linear_layers.cuda()

        output_shape_lin = output_shape if isinstance(output_shape, int) else prod(output_shape)

        self.embedding_head = torch.nn.Linear(
            in_features=internal_embedding_length * (1 + int(form == "parallel")),
            out_features=output_shape_lin,
            bias=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def forward(self, text_embeddings: Union[torch.Tensor, Iterable[torch.Tensor]],
                user_embeddings: torch.Tensor) -> torch.Tensor:
        super(DeepCrossNetwork, self).forward(text_embeddings=text_embeddings, user_embeddings=user_embeddings)

        x_0 = min_max_combine(text_embeddings=text_embeddings, user_embeddings=user_embeddings)

        if self.in_embedding_appendix_module is not None:
            x_0 = torch.concat(
                tensors=(x_0, self.in_embedding_appendix_module(x_0)),
                dim=-1
            )

        cross_embedding = x_0
        for cross_layer in self.cross_layers:
            cross_embedding = cross_layer(x_in=cross_embedding, x_0=x_0)

        if self.form == "stacked":
            ret = self.embedding_head(self.linear_layers(cross_embedding))
        else:
            linear_embedding = self.linear_layers(x_0)
            ret = self.embedding_head(torch.concat(tensors=(cross_embedding, linear_embedding), dim=-1))

        return ret if isinstance(self.output_shape, int) else ret.reshape(
            (text_embeddings.shape[0]
             if isinstance(text_embeddings, torch.Tensor) else
             [text_embedding.shape[0] for text_embedding in text_embeddings][0],
             user_embeddings.shape[0], *self.output_shape)
        )

    def get_embedding_in_lengths(self) -> Tuple[Optional[int], int]:
        return self.text_embedding_length, self.user_embedding_length
