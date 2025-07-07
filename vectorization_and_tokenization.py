# Make sure wordnet, the averaged_perceptron_tagger for the used language, stopwords and punkt_tab is downloaded from nltk; e.g. use:
# >>> nltk.download('wordnet')
# >>> nltk.download('averaged_perceptron_tagger_eng')
# >>> nltk.download('stopwords')
# >>> nltk.download('punkt_tab')

import numpy as np
import scipy.sparse as sp

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



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
STOPWORDS_EN=set(stopwords.words('english'))


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
    
    
    
 ########## TEXT VECTORIZATION ##########   

def vectorize_text_data(df, col_names,vectorization='tf-idf', max_features_per_column=1000, stop_words=None, ngram_range=(1,1), tokenizer=None, language='english', normalizer=None):
    """
    vectorizes text data from columns of a pandas dataframe. Each column will be vectorized separately. 
    
    Parameters:
        - df: dataframe with the data
        - col_names: list; names of the columns to vectorize
        - vectorization='tf-idf'|'bag_of_words', default='tf-idf'; whether tf-idf or bag of words should be used for vectorization
        - max_features_per_column: int|list of int, default=1000; max features to vectorizer, either one integer for all columns or a list of integers corresponding to each column
        - stop_words: "english"|list|None, default=None, whether stopwords should be removed. Use "english" to remove english stopwords or a custom list of stopwords.
        - ngram_range: tuple (min, max); range of n-values for n-grams
        - tokenizer: callable, default=None; custom tokenizer; tokenizer should have argumnets (text, language, remove_stop_words)
        - language: str, default='english'; language of the text data
        - normalizer: callable, default=None, normalization applied to the vectorized data
    
    Returns: 
        - X: vectorized data as numpy array, optionally normalized
        - feature_names: dict; feature names extracted from each column, keys are names of the columns, values the corresponding feature names
    """
    
    X_arrays=[]
    feature_names={}
    if type(max_features_per_column) is int:
        max_features_per_column=[max_features_per_column]*len(col_names)
        
    if tokenizer is not None:
        if stop_words=='english':
            stop_words=STOPWORDS_EN #Use english stopword list from nltk.corpus
        custom_tokenizer=lambda text: tokenizer(text, language=language, stop_words=stop_words) #set up custom tokenizer
    else:
        custom_tokenizer=None
        
    for name, max_features in zip(col_names, max_features_per_column):
        if vectorization=='tf-idf':
            if custom_tokenizer is None:
                vectorizer=TfidfVectorizer(max_features=max_features, stop_words=stop_words, ngram_range=ngram_range)
            else: 
                vectorizer=TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,token_pattern=None, tokenizer=custom_tokenizer)
                
        elif vectorization=='bag_of_words':
            if custom_tokenizer is None:
                vectorizer=CountVectorizer(max_features=max_features, stop_words=stop_words, ngram_range=ngram_range)
            else:
                vectorizer=CountVectorizer(max_features=max_features, ngram_range=ngram_range, token_pattern=None, tokenizer=custom_tokenizer)
        else:
            raise Exception('Invalid vectorization type.')
        
        corpus=list(df[name])
        x_sparse=vectorizer.fit_transform(corpus)
        x=x_sparse.toarray() #Convert to numpy
        
        X_arrays.append(x)
        feature_names[name]=vectorizer.get_feature_names_out()
        
    X=np.hstack(X_arrays)
    if normalizer is not None:
        X=normalizer(X)
    
    return X, feature_names            



class multi_column_vectorizer:
    def __init__(self, col_names,vectorization='tf-idf', max_features_per_column=1000, stop_words=None, ngram_range=(1,1), tokenizer=None, language='english'):
        self.col_names=col_names
        self.vectorization=vectorization

        if type(max_features_per_column) is int or max_features_per_column is None:
            self.max_features_per_column=[max_features_per_column]*len(col_names)
        else:
            self.max_features_per_column=max_features_per_column

        self.ngram_range=ngram_range
        self.tokenizer=None
        self.language=language
        
        if tokenizer is not None:
            if stop_words=='english':
                self.stop_words=STOPWORDS_EN
            else:
                self.stop_words=stop_words #Use english stopword list from nltk.corpus
            self.tokenizer=lambda text: tokenizer(text, language=self.language, stop_words=self.stop_words) #set up custom tokenizer
        else:
            self.stop_words=stop_words
            self.tokenizer=None
        
        self.vectorizers={}

        for name, max_features in zip(self.col_names, self.max_features_per_column):
            if self.vectorization=='tf-idf':
                if self.tokenizer is None:
                    vectorizer=TfidfVectorizer(max_features=max_features, stop_words=self.stop_words, ngram_range=self.ngram_range)
                else: 
                    vectorizer=TfidfVectorizer(max_features=max_features, ngram_range=self.ngram_range,token_pattern=None, tokenizer=self.tokenizer)
                    
            elif vectorization=='bag_of_words':
                if self.tokenizer is None:
                    vectorizer=CountVectorizer(max_features=max_features, stop_words=self.stop_words, ngram_range=self.ngram_range)
                else:
                    vectorizer=CountVectorizer(max_features=max_features, ngram_range=self.ngram_range, token_pattern=None, tokenizer=self.tokenizer)
            else:
                raise Exception('Invalid vectorization type.')
            
            self.vectorizers[name]=vectorizer
        
    def fit_transform(self, df, colnames=None, sparse=True):
        if colnames is None:
            colnames=self.col_names
        X_arrays=[]
        for name in colnames:
            if not name in self.col_names:
                raise Exception(f"No vectorizer for column {name}")
            elif not name in df.columns:
                raise Exception(f"No column named {name} in dataframe")
            
            corpus=list(df[name])
            x=self.vectorizers[name].fit_transform(corpus)
            X_arrays.append(x)
        
        X=sp.hstack(X_arrays)
            
        if not sparse:
            X=X.toarray()

        return X

    def transform(self, df, colnames=None, sparse=True):
        if colnames is None:
            colnames=self.col_names
        X_arrays=[]
        for name in colnames:
            if not name in self.col_names:
                raise Exception(f"No vectorizer for column {name}")
            elif not name in df.columns:
                raise Exception(f"No column named {name} in dataframe")
            
            corpus=list(df[name])
            x=self.vectorizers[name].transform(corpus)
            X_arrays.append(x)
        
        X=sp.hstack(X_arrays)
            
        if not sparse:
            X=X.toarray()

        return X
    
    def get_feature_names(self):
        feature_names={name:self.vectorizers[name].get_feature_names_out() for name in self.col_names}
        return feature_names
            


        


        