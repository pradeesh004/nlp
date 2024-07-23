import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter, defaultdict



nltk.download('punkt')

# input text corpus
text = input("Enter the text :")

#tokenizing
tokens = word_tokenize(text.lower())

#Calulating unigram,bigram,trigram for the given text
unigrams = list(ngrams(tokens, 1))

bigrams = list(ngrams(tokens, 2))

trigrams = list(ngrams(tokens, 3))

# Calculate Bigram Probabilities
bigram_counts = Counter(bigrams)
unigram_counts = Counter(tokens)
bigram_probabilities = {bigram: bigram_counts[bigram] / unigram_counts[bigram[0]] for bigram in bigram_counts}

# Function for Next Word Prediction
def next_word_prediction(prev_word):
    candidates = {bigram: prob for bigram, prob in bigram_probabilities.items() if bigram[0] == prev_word}
    if not candidates:
        return None
    predicted_bigram = max(candidates, key=candidates.get)
    return predicted_bigram[1]

# Print ngam details 
print("Unigrams:")
print(unigrams)
print("\nBigrams:")
print(bigrams)
print("\nTrigrams:")
print(trigrams)
print("\nBigram Probabilities:")
for bigram, prob in bigram_probabilities.items():
    print(f"{bigram}: {prob:.4f}")

# Next word prediction example
previous_word = input("enter the current word to find the next word :")
predicted_word = next_word_prediction(previous_word)
print(f"\nNext word prediction for '{previous_word}': {predicted_word}")
