# ClassifyTweetSentiments
This Python program uses Naive Bayes to classify positive or negative sentiment tweets.

## Input

```data/tweets.csv```: The positive sentiment tweets are indicated with a label of 1, while the negative sentiment tweets are represented with a -1

## Output

Each experiment will have two accuracy variables. The first one is the accuracy of the system without the absent words, and the second is when the absent words are incorporated. The following titles represent each experiment and are printed to the Command Prompt screen.

Accuracy without any stop words<br />
The accuracy when the most frequent 25 words are removed<br />
The accuracy when the most frequent 50 words are removed<br />
The accuracy when the most frequent 100 words are removed<br />
The accuracy when the most frequent 200 words are removed<br />

## Running The Project In Windows

In the project's root directory, type ```pip install -r requirements.txt``` to your Command Prompt to install all dependencies.

Type ```python main.py``` to run the project.

## How Does The Program Work

Every tweet is split into individual words, and the words are stored in a word dictionary or vocabulary. Each word receives a unique id that represents the vocabulary of the dataset. It creates a zero matrix. The number of rows is the vocabulary size, and the number of columns is the number of tweets. This matrix holds the feature vectors of each tweet.

The program then iterates through all words and sets the row inside the feature vector corresponding to the word to one. This method is a form of one-hot encoding where each tweet is a binary vector.

After some zeros in the feature vectors are set to 1, the program calculates the log probabilities of each word appearing in positive and negative sentiments with ```P(word|positive)``` and ```P(word|negative)```. The log probability prevents underflowing. 

Two Naive Bayes classifiers are defined where one uses both the presence and absence of the words while the other only uses the presence of the words. They both take in a tweet and output a sentiment.

Ultimately, the program removes the most frequent words from the vocabulary. The most frequent words are similar to the "stop words," such as "and" and "is." Then a new experiment is performed without the stop words.

