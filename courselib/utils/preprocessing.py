# to use tokenizers, make sure wordnet, the averaged_perceptron_tagger for the used language and punkt_tab is downloaded from nltk; e.g. use:
# >>> nltk.download('wordnet')
# >>> nltk.download('averaged_perceptron_tagger_eng')
# >>> nltk.download('punkt_tab')


import numpy as np
import scipy.sparse as sp

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer

import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .normalization import lp_normalize

def labels_encoding(Y, labels=None, pos_value=1, neg_value=-1):
    """
    Encodes class labels into a one-vs-rest style matrix with custom values.

    Parameters:
    - Y: array-like of shape (N,) – class labels
    - labels: optional list of label values in desired order; if None, inferred from sorted unique values
    - pos_value: value for the positive (true) class (default: 1)
    - neg_value: value for the negative class (default: -1)

    Returns:
    - encoded: ndarray of shape (N, K), where K = number of classes
    """
    Y = np.asarray(Y)
    if labels is None:
        labels = np.unique(Y)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    K = len(labels)
    N = len(Y)
    
    encoded = np.full((N, K), neg_value, dtype=float)
    for i, y in enumerate(Y):
        k = label_to_index[y]
        encoded[i, k] = pos_value

    return encoded

def labels_to_numbers(labels, class_names=None):
    if class_names is None:
        class_names = np.unique(labels)
    label_to_number = {label: i for i, label in enumerate(class_names)}
    return np.array([label_to_number[label] for label in labels])


########################################################################## TOKENIZATION AND VECTORIZATION


def get_wordnet_pos(treebank_tag):
    """Converts treebank_tags obtained e.g. from nltk.pos_tags() to wordnet compatible position tags."""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        # Default to noun if no match is found or starts with 'N'
        return wn.NOUN  


########## PUNCTUATION AND STOPWORD LISTS ############
PUNCTUATIONS=string.punctuation+'’'+ '“'+ '”'



########## TOKENIZERS ############

def basic_word_tokenizer(text, language='english', stop_words=None):
    """
    Basis word tokenizer
    Tokenizes text by words and removes punctuation and lowercases words. Optionally removes given stopwords.
    
    Parameters:
        - text: str; text to tokenize
        - language: str, default='english'; language of the text
         -stop_words: list; list of stopwords to remove
    
    Returns:
        - tokens: list; lowercased words without punctuation
    """
    words=nltk.word_tokenize(text, language=language)
    tokens=[word.lower() for word in words if not word in PUNCTUATIONS] #remove punctuation
    
    if not stop_words is None:
        tokens=[token for token in tokens if not token in stop_words] #remove stopwords
    return tokens
    

def lemmatization_tokenizer(text, language='english', stop_words=None):
    """
    lemmatization tokenizer
    Tokenizes text by words and removes punctuation, lowercases words and applies lemmatization to each word. 
    Optionally removes given stopwords..
    
    Parameters:
        - text: str; text to tokenize
        - language: str, default='english'; language of the text
        - stop_words: list; list of stopwords to remove
    
    Returns:
        - tokens: list; lemmatized lowercased words without punctuation
    """
    
    words=basic_word_tokenizer(text, language=language, stop_words=stop_words)
    words_and_pos=nltk.pos_tag(words)
    tokens=[WordNetLemmatizer().lemmatize(s, get_wordnet_pos(p)) for (s,p) in words_and_pos]
    return tokens

def stemming_tokenizer(text, language='english', stop_words=None):
    """
    stemming tokenizer
    Tokenizes text by words and removes punctuation, lowercases words and applies stemming to each word.
    Optionally removes given stopwords.
    
    Parameters:
        - text: str; text to tokenize
        - language: str, default='english'; language of the text
        - stop_words: list; list of stopwords to remove
    
    Returns:
        -tokens: list; stemmed lowercased words without punctuation
    """

    words=basic_word_tokenizer(text, language=language, stop_words=stop_words)
    stemmer = SnowballStemmer(language)
    tokens=[stemmer.stem(word) for word in words]
    return tokens
    
    
    
 ########## TEXT VECTORIZATER ##########   
 
class multi_column_vectorizer:
    """
    Vectorizer for vectorizing multiple columns of a pandas dataframe. Allows tf-idf as well as bag-of-words vectorization, different ngrams, stop_word_removal, normalization and
    usage of custom tokenizers. 
    
    Attributes:
        - col_names: list of str; names of the colums to vectorize
        - vectorization: 'tf-idf'|'bag-of-words', default='tf-idf; vectorization type
        - max_features_per_column: list of int or None, default=None; max features corresponding to each column,
          if None, the number of features is unlimited
          
        - stop_words: "english"|list|None, default=None, whether stopwords should be removed. Use "english" to remove english stopwords or a custom list of stopwords.
        - ngram_range: tuple (min, max); range of n-values for n-grams
        - tokenizer: callable, default=None; custom tokenizer
        - language: str, default='english'; language of the text data to vectorize (only important when using custom tokenizers)
        - norm_ord: 1|2|None, default=2; order of Lp-normalization to apply
        
        -self.vectorizers: dict of vectorizer, keys are col_names, values vectorizers from sklearn.feature_extraction.text
        
    Methods:
        - fit_transform(df, colnames, sparse): learn vocabulary from columns in colnames of pd dataframe df and vectorizes them, returns a matrix; if sparse=True, this is a sparse matrix, else a numpy array
        - transform(df,colnames, sparse): vectorizes columns in colnames of of pd dataframe df using the learned vocabulary, returns a matrix; if sparse=True, this is a sparse matrix, else a numpy array
        - get_feature_names_out(): gives out the learned vocabularies of each column in dictionary form 
        
        
         
         
    """
    def __init__(self, col_names,vectorization='tf-idf', max_features_per_column=None, stop_words=None, ngram_range=(1,1), tokenizer=None, language='english', normalization='l2'):
        """
        Parameters:
            - col_names: list of str; names of the colums to vectorize
            - vectorization: 'tf-idf'|'bag-of-words', default='tf-idf; vectorization type
            - max_features_per_column: int|list of int|None, default=None; max features either one integer for all columns or a list of integers corresponding to each column,
            if None, the number of features is unlimited
            
            - stop_words: "english"|list|None, default=None, whether stopwords should be removed. Use "english" to remove english stopwords or a custom list of stopwords.
            - ngram_range: tuple (min, max); range of n-values for n-grams
            - tokenizer: callable, default=None; custom tokenizer
            - language: str, default='english'; language of the text data to vectorize (only important when using custom tokenizers)
            - normalization: 'l1'|'l2'|None, default='l2'; wheter to apply L1 or L2 normalization when vectorizing
        """
        
        self.col_names=col_names
        self.vectorization=vectorization
        self.ngram_range=ngram_range
        self.language=language
        
        if normalization=='l2':
            self.norm_ord=2
        elif normalization=='l1':
            self.norm_ord=1
        elif normalization is not None:
            raise Exception(f'Normalization {normalization} not supported.')
        else:
            self.norm_ord=None

        if type(max_features_per_column) is int or max_features_per_column is None:
            self.max_features_per_column=[max_features_per_column]*len(col_names)
        else:
            self.max_features_per_column=max_features_per_column
        
        if tokenizer is not None:
            if stop_words=='english':
               self.stop_words=ENGLISH_STOP_WORDS #Use english stopword list from sklearn.feature_extraction.text
            else:
                self.stop_words=stop_words
            self.tokenizer=lambda text: tokenizer(text, language=self.language, stop_words=self.stop_words) #set up custom tokenizer
        else:
            self.stop_words=stop_words
            self.tokenizer=None
        
    
        #Set up vectorizers
        self.vectorizers={}
        
        #Set up parameters to pass to vectorizers from sklearn.feature_extraction.text below
        args={'ngram_range':self.ngram_range, 'tokenizer':self.tokenizer}
        if tokenizer is not None:
            args['token_pattern']=None # to prevent warning when custom tokenizer is used
            args['stop_words']=None # do not use build in stop word removal from sklearn's vectorizers, as stopword removal should be already handled in custom tokenizers
        else:
            args['stop_words']=self.stop_words
            
        
        for name, max_features in zip(self.col_names, self.max_features_per_column):
            if self.vectorization=='tf-idf':
                vectorizer=TfidfVectorizer(max_features=max_features,  **args)  # although TfidfVectorizer does support a normalization paramter norm, we normalize in fit_transform and transform
                                                                                # for consistency with CountVectorizer which does not directly support normalization
            elif vectorization=='bag-of-words':
                vectorizer=CountVectorizer(max_features=max_features, **args)
            else:
                raise Exception('Invalid vectorization type.')
            
            self.vectorizers[name]=vectorizer
        
    def fit_transform(self, df, col_names=None, sparse=True):
        """
        learn vocabulary from columns in colnames of pd dataframe df and vectorizes them
        
        Parameters:
            - df: pandas dataframe; dataframe containing columns to vectorize
            - colnames: list of str; names of the columns of df to vectorize, should be in self.col_names
            - sparse: bool; if True, output will be a sparse array, else a numpy array
            
        Returns:
            - X array; array containing the vectorized data of all columns
        """
        if col_names is None:
            col_names=self.col_names #vectorize all columns in self.col_names
        X_arrays=[]
        
        for name in col_names:
            #Check if columns exist in self.col_names and df
            if not name in self.col_names:
                raise Exception(f"No vectorizer for column {name}")
            elif not name in df.columns:
                raise Exception(f"No column named {name} in dataframe")
            
            corpus=list(df[name])
            x=self.vectorizers[name].fit_transform(corpus)
            #normalize
            
            if not self.norm_ord is None:
                x=lp_normalize(x, ord=self.norm_ord)
                
            X_arrays.append(x)
        
        X=sp.hstack(X_arrays)
        
        
            
        if not sparse:
            X=X.toarray()

        return X

    def transform(self, df, col_names=None, sparse=True):
        """
        vectorize columns in colnames of pd dataframe df and vectorizes them using the prelearned vocabulary
        
        Parameters:
            - df: pandas dataframe; dataframe containing columns to vectorize
            - colnames: list of str; names of the columns of df to vectorize, should be in self.col_names
            - sparse: bool; if True, output will be a sparse array, else a numpy array
            
        Returns:
            - X array; array containing the vectorized data of all columns
        """
        if col_names is None:
            col_names=self.col_names
        X_arrays=[]
        for name in col_names:
            if not name in self.col_names:
                raise Exception(f"No vectorizer for column {name}")
            elif not name in df.columns:
                raise Exception(f"No column named {name} in dataframe")
            
            corpus=list(df[name])
            x=self.vectorizers[name].transform(corpus)
            
            #normalize
            if not self.norm_ord is None:
                x=lp_normalize(x, ord=self.norm_ord)
            X_arrays.append(x)
        
        X=sp.hstack(X_arrays)
        
            
        if not sparse:
            X=X.toarray()

        return X
    
    def get_feature_names_out(self):
        """
        gives out learned vocabulary for all columns in self.col_names
        
        Returns:
            - feature_names: dict; dictionary with keys the names of the columns (in self.col_names) and values a list of feature names
              (the vocabulary learned from the vectorizer of this column in indexed order)
        """
        feature_names={name:self.vectorizers[name].get_feature_names_out() for name in self.col_names}
        return feature_names
            


        


        
