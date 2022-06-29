from typing import List, Union
import numpy as np
import torch
from numpy import ndarray
from transformers import (AutoTokenizer, AutoModel)

# transformer is a hugging face library that supports thousands of pretrained models
# More about BertTokenizer here https://www.analyticsvidhya.com/blog/2021/09/an-explanatory-guide-to-bert-tokenizer/
class BertEmbedding:

    def __init__(self, gpu_id: int = 0):

        # model: Model is the string path for the bert weights. If given a keyword, the s3 path will be used.
        # custom_model: This is optional if a custom bert model is used.
        # custom_tokenizer: Place to use custom tokenizer.

        # Configuration part related to cpu and gpu
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            assert (
                    isinstance(gpu_id, int) and (0 <= gpu_id and gpu_id < torch.cuda.device_count())
            ), f"`gpu_id` must be an integer between 0 to {torch.cuda.device_count() - 1}. But got: {gpu_id}"

            self.device = torch.device(f"cuda:{gpu_id}")

        # Set the model and the tokenizer
        self.model = AutoModel.from_pretrained("model", output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained("model",do_lower_case=False)

        # The model is set in evaluation mode by default using model.eval() (Dropout modules are deactivated).
        # To train the model, you should first set it back in training mode with model.train().
        #self.model.eval()

    def tokenize_input(self, text: str) -> torch.tensor:
        # text: Text to tokenize.
        # Returns a torch tensor.

        # Split the text to words"tokens" including [CLS] token at the beginning of the sentence and [SEP] at the end
        # It maps the word and its derivatives to the same word as egg and eggs
        tokenized_text = self.tokenizer.tokenize(text)
        # Map the token to its index in the dictionary of words
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert it to torch.tensor ---> a multidimensional matrix
        return torch.tensor([indexed_tokens]).to(self.device)


    def extract_embeddings(self, text: str, hidden: Union[List[int], int] = -1) -> torch.Tensor:
        # text: The text to extract embeddings for.
        # hidden: The hidden layer(s) to use for embeddings.

        tokens_tensor = self.tokenize_input(text)
        # pooled is the output layer embedding of the first token in the sequence
        # hidden_states is the output of all layers each of size: batch_size x sequence_length x embedding state length
        pooled, hidden_states = self.model(tokens_tensor)[-2:]

        # Take only the embeddings of a certain layer and get mean, max or median across the sequence_length
        hidden_s = hidden_states[hidden]
        return hidden_s.mean(dim=1).squeeze()




    # Create the embedding matrix
    def create_matrix(self, content: List[str], hidden: Union[List[int], int] = -1) -> ndarray:

        # content: The list of sentences.
        # hidden: Which hidden layer to use.
        # return: A numpy array matrix of the given content.

        # .cpu:copy the data to cpu, numpy() to create a numpy from a tensor
        # np.squeeze to remove 1 dimensions, .asarray to create a single numpy array from the list of numpy array
        # of each sentence
        return np.asarray([
            np.squeeze(self.extract_embeddings(
                t, hidden=hidden).data.cpu().numpy()) for t in content
        ])

    def __call__(self, content: List[str], hidden: Union[List[int], int] = -1) -> ndarray:

        return self.create_matrix(content, hidden)

