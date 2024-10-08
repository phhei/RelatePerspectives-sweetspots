from typing import Optional, List, Union, Tuple, Iterable, Dict
from typing_extensions import Literal
from itertools import combinations

import torch
import transformers

from transformers.modeling_outputs import SequenceClassifierOutput


from argperspectives.models.RecommenderComponents import UserTowerInterface, CombinerInterface


class Recommender(torch.nn.Module):
    def __init__(
            self,
            text_processing_transformer_tower: str,
            user_tower: UserTowerInterface,
            combiner: CombinerInterface,
            user_ids: Iterable[int],
            class_weights: Optional[Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]] = None,
            freeze_transformer_body: bool = False,
            text_processing_transformer_tower_similar_parameter_strength: Optional[float] = None,
            shared_private_model: Optional[Literal["simple", "expand_loss"]] = None,
            text_processing_tokenizer: bool = True,
            output_dict: bool = True
    ):
        """
        Tailored recommender system for crazy experiments
        :param text_processing_transformer_tower: a module computing a hidden representation of the text
        :param user_tower: a module computing a hidden representation of a user (annotator)
        :param combiner: a module combining the tensors of the towers (user/text)
        :param user_ids: a list of all users/ annotators
        :param class_weights: a tensor/ dict of class weights (for computing the loss) - in general (a tensor) or
        annotator-specific (a dict - you have to provide a tensor for each user/ annotator id)
        (if you don't define anything, each class contributes equally to the loss).
        Each tensor must be 1-dimensional (or n-dimensional if the selected output-shape of the combiner is
        n-dimensional), having scalars for each class. ATTENTION: the scalars will be internally normalized,
        hence a scale of [.1, 1] has a similar effect to [1, 10]
        :param freeze_transformer_body: transformers have their pretrained body and a non-pretrained embedding
        head. Normally, both parts are trained/ weight-adapted, but you can decide to only train the classification head
        :param text_processing_transformer_tower_similar_parameter_strength: activates the Soft-parameter-sharing
        (see Ruder, Sebastian (2017). An Overview of Multi-Task Learning in Deep Neural Networks).
        If you set a value >0, you encourage the modules initializing for parameter sharing to have similar weights.
        If you set the value =0, each user/annotator has his its separate neural net.
        A value of <0 will discourage the modules initializing for parameter sharing to have similar weights.
        :param shared_private_model: introducing a separate private LLM-transformer for each user AND a shared one
        (if soft-parameter-sharing is activated, it affects only the shared LLM)
        (see Liu, Pengfei, Xipeng Qiu, and Xuanjing Huang (July 2017)
        “Adversarial Multitask Learning for Text Classification”).
        If you want to expand the loss function in addition as Jin, N. et al. (Apr. 2020).
        “Multi-Task Learning Model Based on Multi-Scale CNN and LSTM for Sentiment Classification” did,
        write "expand_loss"
        :param text_processing_tokenizer: when you want to handle cases in which the input is raw (plain strings),
        input True here
        :param output_dict: see huggingface-doc: if yes, a SequenceClassifierOutput is returned, else a tuple
        """
        super().__init__()

        self.user_tower: UserTowerInterface = user_tower

        self.text_tower_tokenizer: Optional[transformers.PreTrainedTokenizer] = \
            transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=text_processing_transformer_tower
            ) if text_processing_tokenizer else None
        self.text_tower = torch.nn.ModuleDict()
        self.text_tower_keys: List[str] = []
        if text_processing_transformer_tower_similar_parameter_strength is None:
            self.text_tower_keys.append("shared")
        if text_processing_transformer_tower_similar_parameter_strength is not None:
            self.text_tower_keys.extend(map(lambda ui: f"shared_{ui}", user_ids))
        if shared_private_model is not None:
            self.text_tower_keys.extend(map(lambda ui: str(ui), user_ids))

        for key in self.text_tower_keys:
            llm: transformers.PreTrainedModel = transformers.AutoModel.from_pretrained(
                pretrained_model_name_or_path=text_processing_transformer_tower,
                return_dict=True
            )
            llm.requires_grad_(requires_grad=not freeze_transformer_body)
            llm_embedding_head = torch.nn.Linear(
                in_features=llm.config.hidden_size,
                out_features=self.user_tower.get_embedding_length(),
                bias=True
            )
            self.text_tower[key] = torch.nn.ModuleList(
                [llm, llm_embedding_head]
            )

        if torch.cuda.is_available():
            self.text_tower.cuda()

        self.text_processing_transformer_tower_similar_parameter_strength = \
            text_processing_transformer_tower_similar_parameter_strength
        self.class_weights: Optional[Union[torch.FloatTensor, Dict[int, torch.FloatTensor]]] = class_weights
        if torch.cuda.is_available() and class_weights:
            self.class_weights = \
                self.class_weights.cuda() if isinstance(self.class_weights, torch.Tensor) else \
                    {k: v.cuda() for k, v in self.class_weights.items()}
        self.shared_private_model: Optional[Literal["simple", "expand_loss"]] = shared_private_model
        self.combiner: CombinerInterface = combiner
        self.output_dict: bool = output_dict

    def forward(
            self,
            input_raw: Optional[List[str]] = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            annotator_indecies: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        """
        Computes a forward pass
        :param input_raw: a list of texts which should be processed.
        Works only if text_processing_tokenizer was set to True
        :param input_ids: the tokenized text of shape (#texts, seq-length)
        :param attention_mask: the attention mask of the text of shape (#texts, seq-length)
        :param labels: the labels (indices) of the shape of (#texts, #annotators)
        (or (#texts, #annotators, *output_dims (except the last one)) in case of several output dimensions).
        You can also directly input class probabilities (not recommended), then the shape has to be:
        (#texts, #annotators, *output_dims)
        :param annotator_indecies: the annotators-ids of shape (#annotators, )
        :param return_dict: see huggingface-doc: if yes, a SequenceClassifierOutput is returned, else a tuple
        :return: the prediction (loss and more)
        """

        # compute tensors of shape (#annotators, uniform_embedding_length)
        user_embeddings = self.user_tower(annotator_indecies)

        if input_ids is None:
            if input_raw is None:
                raise AttributeError("You must define input_raw or input_ids, nothing was given!")
            if self.text_tower_tokenizer is None:
                raise RuntimeError("Text-Tokenizing is required but no tokenizer is given!")
            batch_encoding = self.text_tower_tokenizer(
                text=input_raw,
                padding=True,
                return_tensors="pt",
                is_split_into_words=False,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_length=False,
                return_special_tokens_mask=False,
                return_overflowing_tokens=False,
                return_offsets_mapping=False
            )
            input_ids = batch_encoding["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
            attention_mask = batch_encoding["attention_mask"].to("cuda" if torch.cuda.is_available() else "cpu")

        # compute tensors of shape (#texts, uniform_embedding_length)
        text_tower_embeddings: Dict[str, torch.FloatTensor] = {
            k: t[1](t[0](input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :])
            for k, t in self.text_tower.items()
        }

        if self.shared_private_model is None and \
                self.text_processing_transformer_tower_similar_parameter_strength is None:
            prediction = self.combiner(
                text_embeddings=text_tower_embeddings["shared"],
                user_embeddings=user_embeddings
            )
        elif self.shared_private_model is None:
            prediction = torch.stack(
                tensors=[torch.squeeze(self.combiner(
                    text_embeddings=text_tower_embeddings[f"shared_{user_id.item()}"],
                    user_embeddings=torch.unsqueeze(user_embeddings[i, :], dim=0)
                ), dim=1) for i, user_id in enumerate(annotator_indecies)],
                dim=1
            )
        elif self.text_processing_transformer_tower_similar_parameter_strength is None:
            prediction = torch.stack(
                tensors=[torch.squeeze(self.combiner(
                    text_embeddings=(text_tower_embeddings[str(user_id.item())], text_tower_embeddings["shared"]),
                    user_embeddings=torch.unsqueeze(user_embeddings[i, :], dim=0)
                ), dim=1) for i, user_id in enumerate(annotator_indecies)],
                dim=1
            )
        else:
            prediction = torch.stack(
                tensors=[torch.squeeze(self.combiner(
                    text_embeddings=(text_tower_embeddings[str(user_id.item())],
                                     text_tower_embeddings[f"shared_{user_id.item()}"]),
                    user_embeddings=torch.unsqueeze(user_embeddings[i, :], dim=0)
                ), dim=1) for i, user_id in enumerate(annotator_indecies)],
                dim=1
            )

        loss = self.compute_loss(
            logits=prediction,
            hidden_states_of_text_transformers=text_tower_embeddings,
            user_embeddings=user_embeddings,
            labels=labels,
            f_class_weights=
            None if self.class_weights is None else
            (tuple(self.class_weights[_id.item()] for _id in annotator_indecies.cpu())
             if isinstance(self.class_weights, Dict) else self.class_weights)
        )

        if (return_dict is None and self.output_dict) or return_dict:
            return SequenceClassifierOutput(
                loss=loss,
                logits=prediction,
                hidden_states=tuple(emb for emb in text_tower_embeddings.values()),
                attentions=None
            )

        return prediction, loss, tuple(emb for emb in text_tower_embeddings.values()), None

    def compute_loss(
            self,
            logits: torch.Tensor,
            hidden_states_of_text_transformers: Optional[Dict[str, torch.FloatTensor]] = None,
            user_embeddings: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.Tensor] = None,
            f_class_weights: Optional[Union[torch.FloatTensor, Tuple[torch.FloatTensor]]] = None
    ) -> Optional[torch.Tensor]:
        """
        Computes the loss (if labels are given)
        :param logits: the predicted values (#texts, #users, single-prediction_dim0, ..., single-prediction_dimN)
        :param hidden_states_of_text_transformers: {transformer_name: (#texts, uniform_embedding_length)}
        :param user_embeddings: computed user embeddings (from the user-embedding-tower) in shape of
        (#users, uniform_embedding_length)
        :param labels: the ground truth class indices values (#texts, #users)
        (in more-output-dim-cases: single-prediction_dim0, ..., single-prediction_dimN-1) or
        (#texts, #users, single-prediction_dim0, ..., single-prediction_dimN).
         For unknown cases, use class indices and specify class '-1'
        :param f_class_weights: (optional) the class weights which have to be applied for the standard cross entropy
        loss (part). You can either provide general class weights (single-prediction_dim0, ..., single-prediction_dimN)
        for all annotators or for every annotator tensor (in tuple format, #user tensors)
        :return: the loss (averaged)
        """
        if labels is None:
            return None

        assert logits.shape[-1] >= 2, \
            f"Until yet, unary logits (i.e. logits that have to be sigmoided to model a binary class probability " \
            f"scalar between 0 and 1) are not allowed, each output feature has to be modelled with at least " \
            f"two classes (in binary case: C0 for negative class, C1 for positive class) " \
            f"-- invalid shape {logits.shape}"

        flatten_logits = torch.flatten(input=logits, start_dim=0, end_dim=-2)
        if labels.dtype == torch.long:
            flatten_labels = torch.flatten(input=labels)
        else:
            flatten_labels = torch.flatten(input=labels, start_dim=0, end_dim=-2)

        if f_class_weights is None:
            loss = torch.nn.functional.cross_entropy(
                input=flatten_logits, target=flatten_labels,
                ignore_index=-100
            )
        elif isinstance(f_class_weights, torch.Tensor) and isinstance(self.combiner.output_shape, int):
            # simplest case with class weights: uniform class weights + single output dimension
            loss = torch.nn.functional.cross_entropy(
                input=flatten_logits, target=flatten_labels,
                weight=torch.flatten(input=f_class_weights),
                ignore_index=-100
            )
        elif isinstance(f_class_weights, torch.Tensor) and isinstance(self.combiner.output_shape, Tuple):
            # various class dimensions
            flatten_f_class_weights = torch.flatten(input=f_class_weights, start_dim=0, end_dim=-2)
            mixed_formed_logits = torch.flatten(torch.flatten(input=logits, start_dim=0, end_dim=1),
                                                start_dim=1, end_dim=-2)
            mixed_formed_labels = torch.flatten(
                torch.flatten(input=labels, start_dim=0, end_dim=1),
                start_dim=1,
                end_dim=-(1+int(labels.dtype == torch.float))
            )
            loss = torch.mean(
                torch.stack(
                    tensors=[
                        torch.nn.functional.cross_entropy(
                            input=mixed_formed_logits[:, f_class_weights_slice_index],
                            target=mixed_formed_labels[:, f_class_weights_slice_index],
                            weight=flatten_f_class_weights[f_class_weights_slice_index],
                            ignore_index=-100
                        ) for f_class_weights_slice_index in range(flatten_f_class_weights.shape[0])],
                    dim=0
                )
            )
        elif isinstance(f_class_weights, Tuple) and isinstance(self.combiner.output_shape, int):
            # various user class weights
            loss = torch.mean(
                torch.stack(
                    tensors=[
                        torch.nn.functional.cross_entropy(
                            input=logits[:, user_slice_pos],
                            target=labels[:, user_slice_pos],
                            weight=f_class_weights[user_slice_pos],
                            ignore_index=-100
                        ) for user_slice_pos in range(logits.shape[1])
                    ],
                    dim=0
                )
            )
        elif isinstance(f_class_weights, Tuple) and isinstance(self.combiner.output_shape, Tuple):
            # most complex case: various user class weights + various class dimensions
            flatten_f_class_weights = tuple(torch.flatten(input=fcw, start_dim=0, end_dim=-2) for fcw in f_class_weights)
            mixed_formed_logits = torch.flatten(input=logits, start_dim=2, end_dim=-2)
            mixed_formed_labels = torch.flatten(
                input=labels, start_dim=2, end_dim=-1
            ) if labels.dtype == torch.long else torch.flatten(input=labels, start_dim=2, end_dim=-2)
            loss = torch.mean(
                torch.stack(
                    tensors=[torch.stack(
                        tensors=[
                            torch.nn.functional.cross_entropy(
                                input=mixed_formed_logits[:, user_slice_pos, f_class_weights_slice_index],
                                target=mixed_formed_labels[:, user_slice_pos, f_class_weights_slice_index],
                                weight=flatten_f_class_weights[user_slice_pos][f_class_weights_slice_index],
                                ignore_index=-100
                            ) for f_class_weights_slice_index in range(mixed_formed_labels.shape[2])
                        ],
                        dim=0
                    ) for user_slice_pos in range(logits.shape[1])],
                    dim=0
                )
            )
        else:
            raise AttributeError(f"Incompatible types of f_class_weights ({type(f_class_weights)}) and "
                                 f"self.combiner.output_shape({type(self.combiner.output_shape)}). Class weights must "
                                 f"be None are the combination must be (tensor, single-dim), (tensor, multi-dim), "
                                 f"(tuple, single-dim) or (tensor, multi-dim)")

        if self.text_processing_transformer_tower_similar_parameter_strength is not None:
            affected_models = [m for k, m in self.text_tower.items() if k.startswith("shared_")]
            dissimilar_loss = torch.mean(
                torch.stack(
                    tensors=[
                        torch.stack(
                            tensors=[torch.nn.functional.mse_loss(input=v1, target=v2)
                                     for v1, v2 in zip(t1.parameters(), t2.parameters())],
                            dim=0
                        )
                        for t1, t2 in combinations(affected_models, r=2)
                    ],
                    dim=0
                )
            )
            loss = loss + self.text_processing_transformer_tower_similar_parameter_strength * dissimilar_loss

        if self.shared_private_model == "expand_loss" and hidden_states_of_text_transformers is not None:
            # And for avoid feature/ parameter redundancy, Ldiff is for every annotator to add which panellizes a
            # similar output between a private encoder and the shared encoder.
            if "shared" in hidden_states_of_text_transformers:
                l_diff = torch.mean(
                    torch.stack(
                        tensors=[torch.nn.functional.mse_loss(input=private_encoder_embedding,
                                                              target=hidden_states_of_text_transformers["shared"])
                                 for encoder_key, private_encoder_embedding
                                 in hidden_states_of_text_transformers.items() if encoder_key != "shared"],
                        dim=0
                    )
                )
            else:
                l_diff = torch.mean(
                    torch.stack(
                        tensors=[torch.nn.functional.mse_loss(
                            input=hidden_states_of_text_transformers[user_id],
                            target=hidden_states_of_text_transformers[f"shared_{user_id}"]
                        ) for user_id in self.text_tower_keys if not user_id.startswith("shared_")],
                        dim=0
                    )
                )

            loss = loss - l_diff

            if user_embeddings is not None:
                # The user additional adversarial loss is used to train a model to produce shared features such that
                # a classifier cannot reliably predict the
                # user based on these features
                l_adv = torch.mean(
                    torch.stack(
                        tensors=[-torch.nn.functional.cross_entropy(
                            input=torch.flatten(
                                input=self.combiner(text_embeddings=shared_encoder, user_embeddings=user_embeddings),
                                start_dim=0,
                                end_dim=-2
                            ),
                            target=flatten_labels,
                            ignore_index=-100
                        ) for key, shared_encoder in hidden_states_of_text_transformers.items() if "shared" in key]
                    )
                )
                loss = loss + l_adv

        return loss
