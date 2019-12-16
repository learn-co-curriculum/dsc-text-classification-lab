
# Text Classification - Lab

## Introduction

In this lab, we'll use everything we've learned so far to build a model that can classify a text document as one of many possible classes!

## Objectives

You will be able to:

- Perform classification using a text dataset, using sensible preprocessing, tokenization, and feature engineering scheme 
- Use scikit-learn text vectorizers to fit and transform text data into a format to be used in a ML model 



# Getting Started

For this lab, we'll be working with the classic **_Newsgroups Dataset_**, which is available as a training data set in `sklearn.datasets`. This dataset contains many different articles that fall into 1 of 20 possible classes. Our goal will be to build a classifier that can accurately predict the class of an article based on the features we create from the article itself!

Let's get started. Run the cell below to import everything we'll need for this lab. 


```python
import nltk
from nltk.corpus import stopwords
import string
from nltk import word_tokenize, FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
np.random.seed(0)
```

Now, we need to fetch our dataset. Run the cell below to download all the newsgroups articles and their corresponding labels. If this is the first time working with this dataset, scikit-learn will need to download all of the articles from an external repository -- the cell below may take a little while to run. 

The actual dataset is quite large. To save us from extremely long runtimes, we'll work with only a subset of the classes. Here is a list of all the possible classes:

<img src='classes.png'>

For this lab, we'll only work with the following five:

* `'alt.atheism'`
* `'comp.windows.x'`
* `'rec.sport.hockey'`
* `'sci.crypt'`
* `'talk.politics.guns'`

In the cell below:

* Create a list called `categories` that contains the five newsgroups classes listed above, as strings 
* Get the training set by calling `fetch_20newsgroups()` and passing in the following parameters:
    * `subset='train'`
    * `categories=categories`
    * `remove=('headers', 'footers', 'quotes')` -- this is so that the model can't overfit to metadata included in the articles that sometimes acts as a dead-giveaway as to what class the article belongs to  
* Get the testing set as well by passing in the same parameters, with the exception of `subset='test` 


```python
categories = ['alt.atheism', 'comp.windows.x', 'rec.sport.hockey', 'sci.crypt', 'talk.politics.guns']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
```

    Downloading 20news dataset. This may take a few minutes.
    Downloading dataset from https://ndownloader.figshare.com/files/5975967 (14 MB)


Great! Let's break apart the data and the labels, and then inspect the class names to see what the actual newsgroups are.

In the cell below:

* Grab the data from `newsgroups_train.data` and store it in the appropriate variable  
* Grab the labels from `newsgroups_train.target` and store it in the appropriate variable  
* Grab the label names from `newsgroups_train.target_names` and store it in the appropriate variable  
* Display the `label_names` so that we can see the different classes of articles that we're working with, and confirm that we grabbed the right ones  


```python
data = newsgroups_train.data
target = newsgroups_train.target
label_names = newsgroups_train.target_names
label_names
```




    ['alt.atheism',
     'comp.windows.x',
     'rec.sport.hockey',
     'sci.crypt',
     'talk.politics.guns']



Finally, let's check the shape of `data` to see what our data looks like. We can do this by checking the `.shape` attribute of `newsgroups_train.filenames`.

Do this now in the cell below.


```python
newsgroups_train.filenames.shape
```




    (2814,)



Our dataset contains 2,814 different articles spread across the five classes we chose. 

### Cleaning and Preprocessing Our Data

Now that we have our data, the fun part begins. We'll need to begin by preprocessing and cleaning our text data. As you've seen throughout this section, preprocessing text data is a bit more challenging that working with more traditional data types because there's no clear-cut answer for exactly what sort of preprocessing and cleaning we need to do. Before we can begin cleaning and preprocessing our text data, we need to make some decisions about things such as:

* Do we remove stop words or not?
* Do we stem or lemmatize our text data, or leave the words as is?
* Is basic tokenization enough, or do we need to support special edge cases through the use of regex?
* Do we use the entire vocabulary, or just limit the model to a subset of the most frequently used words? If so, how many?
* Do we engineer other features, such as bigrams, or POS tags, or Mutual Information Scores?
* What sort of vectorization should we use in our model? Boolean Vectorization? Count Vectorization? TF-IDF? More advanced vectorization strategies such as Word2Vec?


These are all questions that we'll need to think about pretty much anytime we begin working with text data. 

Let's get right into it. We'll start by getting a list of all of the english stopwords, and concatenating them with a list of all the punctuation. 

In the cell below:

* Get all the english stopwords from `nltk` 
* Get all of the punctuation from `string.punctuation`, and convert it to a list 
* Add the two lists together. Name the result `stopwords_list` 
* Create another list containing various types of empty strings and ellipses, such as `["''", '""', '...', '``']`. Add this to our `stopwords_list`, so that we won't have tokens that are only empty quotes and such  


```python
stopwords_list = stopwords.words('english') + list(string.punctuation)
stopwords_list += ["''", '""', '...', '``']
```

Great! We'll leave these alone for now, until we're ready to remove stop words after the tokenization step. 

Next, let's try tokenizing our dataset. In order to save ourselves some time, we'll write a function to clean our dataset, and then use Python's built-in `map()` function to clean every article in the dataset at the same time. 

In the cell below, complete the `process_article()` function. This function should:

* Take in one parameter, `article` 
* Tokenize the article using the appropriate function from `nltk` 
* Lowercase every token, remove any stopwords found in `stopwords_list` from the tokenized article, and return the results 


```python
def process_article(article):
    tokens = nltk.word_tokenize(article)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in stopwords_list]
    return stopwords_removed    
```

Now that we have this function, let's go ahead and preprocess our data, and then move into exploring our dataset. 

In the cell below:

* Use Python's `map()` function and pass in two parameters: the `process_article` function and the `data`. Make sure to wrap the whole map statement in a `list()`.

**_Note:_** Running this cell may take a minute or two!


```python
processed_data = list(map(process_article, data))
```

Great. Now, let's inspect the first article in `processed_data` to see how it looks. 

Do this now in the cell below.


```python
processed_data[0]
```




    ['note',
     'trial',
     'updates',
     'summarized',
     'reports',
     '_idaho',
     'statesman_',
     'local',
     'nbc',
     'affiliate',
     'television',
     'station',
     'ktvb',
     'channel',
     '7',
     'randy',
     'weaver/kevin',
     'harris',
     'trial',
     'update',
     'day',
     '4',
     'friday',
     'april',
     '16',
     '1993',
     'fourth',
     'day',
     'trial',
     'synopsis',
     'defense',
     'attorney',
     'gerry',
     'spence',
     'cross-examined',
     'agent',
     'cooper',
     'repeated',
     'objections',
     'prosecutor',
     'ronald',
     'howen',
     'spence',
     'moved',
     'mistrial',
     'denied',
     'day',
     'marked',
     'caustic',
     'cross-examination',
     'deputy',
     'marshal',
     'larry',
     'cooper',
     'defense',
     'attorney',
     'gerry',
     'spence',
     'although',
     'spence',
     'explicitly',
     'stated',
     'one',
     'angle',
     'stategy',
     'must',
     'involve',
     'destroying',
     'credibility',
     'agent',
     'cooper',
     'cooper',
     'government',
     "'s",
     'eyewitness',
     'death',
     'agent',
     'degan',
     'spence',
     'attacked',
     'cooper',
     "'s",
     'credibility',
     'pointing',
     'discrepancies',
     'cooper',
     "'s",
     'statements',
     'last',
     'september',
     'made',
     'court',
     'cooper',
     'conceded',
     'things',
     'compressed',
     'seconds',
     "'s",
     'difficult',
     'remember',
     'went',
     'first',
     'cooper',
     'acknowledged',
     'carried',
     '9mm',
     'colt',
     'commando',
     'submachine',
     'gun',
     'silenced',
     'barrel',
     'thought',
     'colt',
     'commando',
     'revolver',
     'cooper',
     'continued',
     'stating',
     'federal',
     'agents',
     'specific',
     'plans',
     'use',
     'weapon',
     'started',
     'kill',
     'weaver',
     "'s",
     'dog',
     'spence',
     'asked',
     'seven',
     'cartridges',
     'could',
     'fired',
     "degan's",
     'm-16',
     'rifle',
     'degan',
     'apparently',
     'dead',
     'cooper',
     'could',
     'say',
     'sure',
     'degan',
     'return',
     'fire',
     'going',
     'spence',
     'continued',
     'asking',
     'many',
     'agents',
     'extent',
     'cooper',
     'discussed',
     'last',
     'august',
     "'s",
     'events',
     'cooper',
     'responded',
     "'re",
     'implying',
     'got',
     'story',
     'together',
     "'re",
     'wrong',
     'counselor',
     'spence',
     'continued',
     'advance',
     'defense',
     "'s",
     'version',
     'events',
     'namely',
     'marshal',
     'started',
     'shooting',
     'killing',
     'weaver',
     "'s",
     'dog',
     'cooper',
     'disagreed',
     'assistant',
     'u.s.',
     'attorney',
     'ronald',
     'howen',
     'repeatedly',
     'objected',
     "spence's",
     'virulent',
     'cross-examination',
     'agent',
     'cooper',
     'arguing',
     'questions',
     'repetitive',
     'spence',
     'wasting',
     'time',
     'howen',
     'also',
     'complained',
     'spence',
     'improperly',
     'using',
     'cross-examination',
     'advance',
     'defense',
     "'s",
     'version',
     'events',
     'u.s.',
     'district',
     'judge',
     'edward',
     'lodge',
     'sustained',
     'many',
     'objections',
     'however',
     'lawyers',
     'persisted',
     'judge',
     'lodge',
     'jury',
     'leave',
     'room',
     'proceded',
     'admonish',
     'attorneys',
     "'m",
     'going',
     'play',
     'games',
     'either',
     'counsel',
     'personality',
     'problem',
     'day',
     '1',
     'start',
     'acting',
     'like',
     'professionals',
     'spence',
     'told',
     'judge',
     'evidence',
     "'ll",
     'see',
     'agent',
     'larry',
     'cooper',
     'testimony',
     'credible',
     'panicked',
     'remember',
     'sequence',
     'events',
     'spence',
     'continued',
     "'re",
     'going',
     'find',
     'unlikely',
     'similarity',
     'almost',
     'come',
     'cookie',
     'cutter',
     'testimony',
     'mr.',
     'cooper',
     'witnesses',
     'spence',
     'moved',
     'mistrial',
     'grounds',
     'howen',
     "'s",
     'repeated',
     'objections',
     'would',
     'prevent',
     'fair',
     'trial',
     'ca',
     "n't",
     'fair',
     'trial',
     'jury',
     'believes',
     "'m",
     'sort',
     'charlatan',
     'jury',
     'believes',
     "i'm",
     'bending',
     'rules',
     'engaging',
     'delaying',
     'tactic',
     "i'm",
     'violating',
     'court',
     'orders',
     'judge',
     'lodge',
     'called',
     'notion',
     'repeated',
     'sustainings',
     "howen's",
     'objections',
     'somehow',
     'prejudiced',
     'jury',
     'preposterous',
     'denied',
     'motion',
     'mistrial',
     'lodge',
     'tell',
     'howen',
     'restrict',
     'comments',
     'objecting',
     'trial',
     'resumed',
     'prosecution',
     'calling',
     'fbi',
     'special',
     'agent',
     'greg',
     'rampton',
     'prosecution',
     "'s",
     'purpose',
     'simply',
     'introduce',
     'five',
     'weapons',
     'found',
     'cabin',
     'evidence',
     'however',
     'defense',
     'seized',
     'opportunity',
     'address',
     'cooper',
     "'s",
     'credibility',
     'defense',
     'attorney',
     'ellison',
     'matthews',
     'harris',
     'attorney',
     'questioned',
     'rampton',
     'dog',
     'rampton',
     'stated',
     'specific',
     'plans',
     'kill',
     'weaver',
     "'s",
     'dog',
     'without',
     'detected',
     'matthews',
     'rampton',
     'read',
     'septtember',
     '15',
     '1992',
     'transcript',
     'rampton',
     'said',
     'cooper',
     'said',
     'purpose',
     'silenced',
     'weapon',
     'kill',
     'dog',
     'without',
     'detected',
     'dog',
     'chased',
     'rampton',
     'acknowledged',
     'believed',
     'cooper',
     'said',
     'could',
     'remember',
     'stated',
     'conduct',
     'primary',
     'interview',
     'deputy',
     'cooper',
     'conversations',
     'since',
     'interview',
     'conducted']



Now, let's move onto exploring the dataset a bit more. Let's start by getting the total vocabulary size of the training dataset. We can do this by creating a `set` object and then using it's `.update()` method to iteratively add each article. Since it's a set, it will only contain unique words, with no duplicates. 

In the cell below:

* Create a `set()` object called `total_vocab` 
* Iterate through each tokenized article in `processed_data` and add it to the set using the set's `.update()` method 
* Once all articles have been added, get the total number of unique words in our training set by taking the length of the set 


```python
total_vocab = set()
for comment in processed_data:
    total_vocab.update(comment)
len(total_vocab)
```




    46990



### Exploring Data With Frequency Distributions

Great -- our processed dataset contains 46,990 unique words! 

Next, let's create a frequency distribution to see which words are used the most! 

In order to do this, we'll need to concatenate every article into a single list, and then pass this list to `FreqDist()`. 

In the cell below:

* Create an empty list called `articles_concat` 
* Iterate through `processed_data` and add every article it contains to `articles_concat` 
* Pass `articles_concat` as input to `FreqDist()`  
* Display the top 200 most used words  


```python
articles_concat = []
for article in processed_data:
    articles_concat += article
```


```python
articles_freqdist = FreqDist(articles_concat)
articles_freqdist.most_common(200)
```




    [('--', 29501),
     ('x', 4840),
     ("'s", 3203),
     ("n't", 2933),
     ('1', 2529),
     ('would', 1985),
     ('0', 1975),
     ('one', 1758),
     ('2', 1664),
     ('people', 1243),
     ('use', 1146),
     ('get', 1068),
     ('like', 1036),
     ('file', 1024),
     ('3', 1005),
     ('also', 875),
     ('key', 869),
     ('4', 864),
     ('could', 853),
     ('know', 814),
     ('think', 814),
     ('time', 781),
     ('may', 729),
     ('even', 711),
     ('new', 706),
     ('first', 678),
     ('*/', 674),
     ('system', 673),
     ('5', 673),
     ('well', 670),
     ('information', 646),
     ('make', 644),
     ('right', 638),
     ('see', 636),
     ('many', 634),
     ('two', 633),
     ('/*', 611),
     ('good', 608),
     ('used', 600),
     ('7', 593),
     ('government', 588),
     ('way', 572),
     ('available', 568),
     ('window', 568),
     ("'m", 562),
     ('db', 553),
     ('much', 540),
     ('encryption', 537),
     ('6', 537),
     ('using', 527),
     ('say', 523),
     ('gun', 520),
     ('number', 518),
     ('program', 515),
     ('us', 510),
     ('team', 498),
     ('must', 483),
     ('law', 476),
     ('since', 449),
     ('need', 444),
     ('game', 439),
     ('chip', 437),
     ('something', 435),
     ('8', 427),
     ('want', 421),
     ('god', 419),
     ('server', 417),
     ("'ve", 416),
     ('public', 408),
     ('year', 401),
     ('set', 396),
     ('ca', 392),
     ('find', 391),
     ('please', 386),
     ('point', 385),
     ('without', 383),
     ('n', 383),
     ('might', 381),
     ('read', 378),
     ('said', 378),
     ('believe', 378),
     ('go', 377),
     ('take', 377),
     ('really', 376),
     ('version', 374),
     ('c', 374),
     ('anyone', 371),
     ('second', 370),
     ('list', 367),
     ('code', 367),
     ('another', 362),
     ('keys', 362),
     ("'re", 361),
     ('work', 360),
     ('example', 359),
     ('clipper', 358),
     ('play', 357),
     ('problem', 356),
     ('things', 353),
     ('data', 353),
     ('made', 348),
     ('widget', 345),
     ('sure', 344),
     ('however', 344),
     ('case', 343),
     ('still', 342),
     ('back', 341),
     ('entry', 341),
     ('hockey', 340),
     ('last', 339),
     ('10', 339),
     ("'d", 335),
     ('let', 333),
     ('better', 332),
     ('25', 331),
     ('part', 330),
     ('security', 327),
     ('output', 327),
     ('probably', 324),
     ('subject', 322),
     ('line', 321),
     ('privacy', 321),
     ('question', 320),
     ('going', 319),
     ('period', 315),
     ('state', 312),
     ('course', 311),
     ('name', 311),
     ('anonymous', 307),
     ('9', 303),
     ('years', 302),
     ('look', 301),
     ('files', 300),
     ('got', 299),
     ('true', 299),
     ('control', 298),
     ('fact', 294),
     ('long', 293),
     ('application', 291),
     ('every', 290),
     ('season', 290),
     ("'ll", 289),
     ('someone', 285),
     ('source', 284),
     ('possible', 283),
     ('help', 282),
     ('message', 280),
     ('55.0', 279),
     ('games', 276),
     ('thing', 276),
     ('never', 275),
     ('following', 274),
     ('send', 273),
     ('try', 271),
     ('best', 270),
     ('motif', 270),
     ('general', 269),
     ('email', 269),
     ('run', 269),
     ('rather', 268),
     ('actually', 265),
     ('several', 264),
     ('thanks', 264),
     ('means', 264),
     ('either', 263),
     ('give', 263),
     ('note', 262),
     ('keep', 262),
     ('little', 262),
     ('put', 262),
     ('different', 261),
     ('guns', 259),
     ('enough', 259),
     ('given', 256),
     ('far', 255),
     ('come', 254),
     ('group', 253),
     ('seems', 252),
     ('around', 250),
     ('person', 249),
     ('order', 249),
     ('call', 248),
     ('next', 246),
     ('support', 246),
     ('anything', 245),
     ('least', 244),
     ('e', 242),
     ('section', 240),
     ('internet', 238),
     ('power', 236),
     ('open', 235),
     ('sun', 235),
     ('etc', 234),
     ('world', 233),
     ('user', 231),
     ('mail', 231),
     ('rights', 229),
     ('great', 229),
     ('real', 229),
     ('nhl', 227)]



At first glance, none of these words seem very informative -- for most of the words represented here, it would be tough to guess if a given word is used equally among all five classes, or is disproportionately represented among a single class. This makes sense, because this frequency distribution represents all the classes combined. This tells us that these words are probably the least important, as they are most likely words that are used across multiple classes, thereby providing our model with little actual signal as to what class they belong to. This tells us that we probably want to focus on words that appear heavily in articles from a given class, but rarely appear in articles from other classes. You may recall from previous lessons that this is exactly where **_TF-IDF Vectorization_** really shines!

### Vectorizing with TF-IDF

Although NLTK does provide functionality for vectorizing text documents with TF-IDF, we'll make use of scikit-learn's TF-IDF vectorizer, because we already have experience with it, and because it's a bit easier to use, especially when the models we'll be feeding the vectorized data into are from scikit-learn, meaning that we don't have to worry about doing any extra processing to ensure they play nicely together. 

Recall that in order to use scikit-learn's `TfidfVectorizer()`, we need to pass in the data as raw text documents -- the `TfidfVectorizer()` handles the count vectorization process on it's own, and then fits and transforms the data into TF-IDF format. 

This means that we need to:

* Import `TfidfVectorizer` from `sklearn.feature_extraction.text` and instantiate `TfidfVectorizer()` 
* Call the vectorizer object's `.fit_transform()` method and pass in our `data` as input. Store the results in `tf_idf_data_train` 
* Also create a vectorized version of our testing data, which can be found in `newsgroups_test.data`. Store the results in `tf_idf_data_test`. 


**_NOTE:_** When transforming the test data, use the `.transform()` method, not the `.fit_transform()` method, as the vectorizer has already been fit to the training data. 


```python
from sklearn.feature_extraction.text import TfidfVectorizer
```


```python
vectorizer = TfidfVectorizer()
```


```python
tf_idf_data_train = vectorizer.fit_transform(data)
```


```python
tf_idf_data_test = vectorizer.transform(newsgroups_test.data)
```

### Modeling Our Data

Great! We've now preprocessed and explored our dataset, let's take a second to see what our data looks like in vectorized form. 

In the cell below, get the shape of `tf_idf_data`.


```python
tf_idf_data_train.shape
```




    (2814, 36622)



Our vectorized data contains 2,814 articles, with 36,622 unique words in the vocabulary. However, the vast majority of these columns for any given article will be zero, since every article only contains a small subset of the total vocabulary. Recall that vectors mostly filled with zeros are referred to as **_Sparse Vectors_**. These are extremely common when working with text data. 

Let's check out the average number of non-zero columns in the vectors. Run the cell below to calculate this average. 


```python
non_zero_cols = tf_idf_data_train.nnz / float(tf_idf_data_train.shape[0])
print("Average Number of Non-Zero Elements in Vectorized Articles: {}".format(non_zero_cols))

percent_sparse = 1 - (non_zero_cols / float(tf_idf_data_train.shape[1]))
print('Percentage of columns containing 0: {}'.format(percent_sparse))
```

    Average Number of Non-Zero Elements in Vectorized Articles: 107.28038379530916
    Percentage of columns containing 0: 0.9970706028126451


As we can see from the output above, the average vectorized article contains 107 non-zero columns. This means that 99.7% of each vector is actually zeroes! This is one reason why it's best not to create your own vectorizers, and rely on professional packages such as scikit-learn and NLTK instead -- they contain many speed and memory optimizations specifically for dealing with sparse vectors. This way, we aren't wasting a giant chunk of memory on a vectorized dataset that only has valid information in 0.3% of it. 

Now that we've vectorized our dataset, let's create some models and fit them to our vectorized training data. 

In the cell below:

* Instantiate `MultinomialNB()` and `RandomForestClassifier()`. For random forest, set `n_estimators` to `100`. Don't worry about tweaking any of the other parameters  
* Fit each to our vectorized training data 
* Create predictions for our training and test sets
* Calculate the `accuracy_score()` for both the training and test sets (you'll find our training labels stored within the variable `target`, and the test labels stored within `newsgroups_test.target`) 


```python
nb_classifier = MultinomialNB()
rf_classifier = RandomForestClassifier(n_estimators=100)
```


```python
nb_classifier.fit(tf_idf_data_train, target)
nb_train_preds = nb_classifier.predict(tf_idf_data_train)
nb_test_preds = nb_classifier.predict(tf_idf_data_test)
```


```python
rf_classifier.fit(tf_idf_data_train, target)
rf_train_preds = rf_classifier.predict(tf_idf_data_train)
rf_test_preds = rf_classifier.predict(tf_idf_data_test)
```


```python
nb_train_score = accuracy_score(target, nb_train_preds)
nb_test_score = accuracy_score(newsgroups_test.target, nb_test_preds)
rf_train_score = accuracy_score(target, rf_train_preds)
rf_test_score = accuracy_score(newsgroups_test.target, rf_test_preds)

print("Multinomial Naive Bayes")
print("Training Accuracy: {:.4} \t\t Testing Accuracy: {:.4}".format(nb_train_score, nb_test_score))
print("")
print('-'*70)
print("")
print('Random Forest')
print("Training Accuracy: {:.4} \t\t Testing Accuracy: {:.4}".format(rf_train_score, rf_test_score))
```

    Multinomial Naive Bayes
    Training Accuracy: 0.9531 		 Testing Accuracy: 0.8126
    
    ----------------------------------------------------------------------
    
    Random Forest
    Training Accuracy: 0.9851 		 Testing Accuracy: 0.7896


### Interpreting Results

**_Question:_** Interpret the results seen above. How well did the models do? How do they compare to random guessing? How would you describe the quality of the model fit?

Write your answer below:


```python
"""
The models did well. Since there are five classes, the naive accuracy rate (random guessing) would be 20%. 
With scores of 78 and 81 percent, the models did much better than random guessing. 
There is some evidence of overfitting, as the scores on the training set are much higher than those of the test set. 
This suggests that the models' fits could be improved with some tuning.
"""
```

# Summary

In this lab, we used our NLP skills to clean, preprocess, explore, and fit models to text data for classification. This wasn't easy -- great job!!
