from functools import partial
from typing import List, Optional, Union

from transformers import (PreTrainedModel, PreTrainedTokenizer)

from summary_processor import SummaryProcessor
from sentence_handler import SentenceHandler
from bert_embedding import BertEmbedding


class BertSummarizer(SummaryProcessor):
    # Summarizer based on the BERT model

    def __init__(
        self,
        hidden: Union[List[int], int] = -1,
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345,
        gpu_id: int = 0,
    ):


        # hidden: This signifies which layer(s) of the BERT model you would like to use as embeddings.
        # reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        # sentence_handler: The handler to process sentences.
        # random_state: The random state to reproduce summarizations.
        # hidden_concat: Whether or not to concat multiple hidden layers.
        # gpu_id: GPU device index if CUDA is available.

        model = BertEmbedding(gpu_id)
        model_func = partial(model, hidden=hidden)
        super().__init__(model_func, sentence_handler, random_state)

