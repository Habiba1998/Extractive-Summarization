from typing import List
from spacy.lang.ar import Arabic
from spacy.language import Language
from summarizer.text_processors.sentence_abc import SentenceABC

class SentenceHandler(SentenceABC):

    def __init__(self, language: Language = Arabic):
        """
        # Base Sentence Handler with Spacy support.
        # language: Determines the language to use with spacy.
        """
        nlp = language()

        is_spacy_3 = False
        # Create spacy document that can
        try:
            # Supports spacy 2.0
            nlp.add_pipe(nlp.create_pipe('sentencizer'))
        except Exception:
            # Supports spacy 3.0
            nlp.add_pipe("sentencizer")
            is_spacy_3 = True

        # Initialize the parent sentence handler
        super().__init__(nlp, is_spacy_3)

    # Create scapy document from the string "text to be summarized" and pass it to sentence_processor function to return list
    # of sentences
    def process(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        doc = self.nlp(body)
        return self.sentence_processor(doc, min_length, max_length)