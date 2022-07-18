from typing import List
from spacy.lang.ar import Arabic
from spacy.language import Language


class SentenceHandler():

    def __init__(self, language: Language = Arabic):
        """
        # Base Sentence Handler with Spacy support.
        # language: Determines the language to use with spacy.
        """
        self.nlp = language()

        self.is_spacy_3 = False
        # Create spacy document that can
        try:
            # Supports spacy 2.0
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        except Exception:
            # Supports spacy 3.0
            self.nlp.add_pipe("sentencizer")
            self.is_spacy_3 = True

       


   
    def sentence_processor(
            self, doc, min_length: int = 40, max_length: int = 600) -> List[str]:

        to_return = []

        for c in doc.sents:
            if max_length > len(c.text.strip()) > min_length:

                if self.is_spacy_3:
                    to_return.append(c.text.strip())
                else:
                    to_return.append(c.string.strip())

        return to_return

    # Create scapy document from the string "text to be summarized" and pass it to sentence_processor function to return list
    # of sentences
    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        doc = self.nlp(body)
        return self.sentence_processor(doc, min_length, max_length)

    def __call__(
            self, body: str, min_length: int = 40, max_length: int = 600
    ) -> List[str]:
        return self.process(body, min_length, max_length)
