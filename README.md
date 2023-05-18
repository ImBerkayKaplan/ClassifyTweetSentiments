# ClassifyTweetSentiments
Several techniques exist to classify certain sentences based on specific cues. In this project, Naive Bayes will be used to classify example tweets as negative or positive sentiment. The positive sentiment tweets are indicated with a label of 1, while the negative sentiment tweets are represented with -1 in the ```tweets.csv``` file. In addition, most frequent words from the tweets will be removed to demonstrate the negative effects on the accuracy rate.

## Setup

Certain modules must be installed before execution. The list of modules are: 
* numpy
* pandas
* sklearn

To install numpy, use the ```pip install numpy``` command.
To install pandas, use the ```pip install pandas``` command.
To install sklearn, use the ```pip install scikit-learn``` command.

## Execution

To run, simply enter ```python main.py``` into the command window while you're in the project directory and the ```tweets.csv``` is present in the same folder where ```main.py``` exists. The results will be printed out to the command window.
