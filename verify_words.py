import nltk
from spellchecker import SpellChecker

# Download the NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize



def is_word_correct(word):
        # Tokenize the word
        tokens = word_tokenize(word)
        
        for token in tokens:
            # Perform spellchecking for each token
            if not is_token_correct(token):
                return False
        
        return True

def is_token_correct(token):
        # Get the part-of-speech tag for the token
        pos_tag = nltk.pos_tag([token])[0][1]
        
        # Check if the token is a word or a punctuation mark
        if pos_tag.startswith('NN') or pos_tag.startswith('JJ') or pos_tag.startswith('VB') or pos_tag.startswith('RB'):
            # If the token is a word, check if it exists in WordNet
            if not wordnet.synsets(token):
                return False
        
        return True


# Example usage
word = "hi"  # Misspelled word




is_correct = is_word_correct(word)
if not is_correct:
    spell = SpellChecker()
    corrected_text = spell.correction(word)
    print(corrected_text)  # Output: "spelling"

     
   
