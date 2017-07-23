# postag

This is an experiment to see how well a unigram [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) can learn part-of-speech tagging.

# Data

The data comes from [here](http://www.cnts.ua.ac.be/conll2000/chunking/). I ignore the chunking data and focus only on the individual parts of speech.

The model requires GloVe word embeddings, which can be trained with something like [tweetembed](https://github.com/unixpickle/tweetembed).
