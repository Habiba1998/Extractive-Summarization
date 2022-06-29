from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from cluster_features import ClusterFeatures
from sentence_handler import SentenceHandler

AGGREGATE_MAP = {
    'mean': np.mean,
    'min': np.min,
    'median': np.median,
    'max': np.max,
}

class SummaryProcessor:
    #General Summarizer Parent for all clustering processing

    def __init__(self, model: Callable, sentence_handler: SentenceHandler, random_state: int = 12345):

        # model: The callable model for creating embeddings from sentences.
        # sentence_handler: The handler to process sentences. If want to use coreference, instantiate and pass.
        # random_state: The random state to reproduce summarizations: used in initializing clusters

        np.random.seed(random_state)
        self.model = model
        self.sentence_handler = sentence_handler
        self.random_state = random_state


    # Get the best number of clusters
    def calculate_optimal_k(self, body: str, min_length: int = 40, max_length: int = 600, k_max: int = None,) -> int:

        # As the previous function but it returns The optimal k value as an int.

        sentences = self.sentence_handler(body, min_length, max_length)

        if k_max is None:
            k_max = len(sentences) - 1

        hidden = self.model(sentences)
        optimal_k = ClusterFeatures(
            hidden, random_state=self.random_state).calculate_optimal_cluster(k_max)

        return optimal_k

    # Get the sentences and embeddings of the summary
    def cluster_runner(self, sentences: List[str], ratio: float = 0.2,
                       num_sentences: int = 3,) -> Tuple[List[str], np.ndarray]:

        # sentences: Content list of sentences.
        # ratio: The ratio to use for clustering.
        # num_sentences: Number of sentences to use for summarization.
        # return: A tuple of summarized sentences and embeddings


        hidden = self.model(sentences)

        summary_sentence_indices = ClusterFeatures(
            hidden, random_state=self.random_state).cluster(ratio, num_sentences)

        sentences = [sentences[j] for j in summary_sentence_indices]
        embeddings = np.asarray([hidden[j] for j in summary_sentence_indices])

        return sentences, embeddings



    # Get the summary as list of sentences or a paragraph
    def run(self, body: str, ratio: float = 0.2, min_length: int = 40, max_length: int = 600,
            num_sentences: int = None, return_as_list: bool = False) -> Union[List, str]:

        #Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.
        # body: The raw string body to process
        # ratio: Ratio of sentences to use
        # min_length: Minimum length of sentence candidates to utilize for the summary.
        # max_length: Maximum length of sentence candidates to utilize for the summary
        # num_sentences: Number of sentences to use (overrides ratio).
        # return_as_list: Whether or not to return sentences as list.
        # return: A summary sentence

        sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            #num_sentences = self.calculate_optimal_k(body)
            sentences, _ = self.cluster_runner(sentences, ratio, num_sentences)

        if return_as_list:
            return sentences
        else:
            for i in range(len(sentences)):
                sentences[i] = sentences[i].replace("\n", " ")
            return ' '.join(sentences)
