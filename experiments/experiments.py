#Import required packages
import sys,os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kagglehub
from courselib.utils.preprocessing import multi_column_vectorizer, basic_word_tokenizer, lemmatization_tokenizer, stemming_tokenizer
from courselib.utils.metrics import binary_accuracy, accuracy
from courselib.models.glm import LogisticRegression
from courselib.models.svm import LinearSVM
from courselib.models.nn import MLP
from courselib.optimizers import GDOptimizer
import time
from IPython.display import clear_output, display
import scipy.sparse as sp
from courselib.utils.normalization import standardize, standardize_sparse_matrix
from courselib.utils.splits import train_test_split
from courselib.utils.preprocessing import labels_encoding


ACCURACIES={'LogisticRegression': lambda y_pred, y_true: binary_accuracy(y_pred=y_pred, y_true=y_true, class_labels=[0,1]),
             'LinearSVM': lambda y_pred, y_true: binary_accuracy(y_pred=y_pred, y_true=y_true, class_labels=[-1,1]),
               'MLP': accuracy}


def load_data_and_split(training_data_fraction):
    #Download latest version of dataset
    print("Load or download dataset...")
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset") #path to downloaded dataset
    #   (if already downloaded, will not download again)

    fake_path=os.path.join(path, "Fake.csv")    #path to dataset with true news
    true_path=os.path.join(path, "True.csv")    #path to dataset with fake news

    #Read into dataframes
    print("Loading fake.csv ...")
    fake_df=pd.read_csv(fake_path)
    print("Loading true.csv ...")
    true_df=pd.read_csv(true_path)

    #Label data (1=true, 0=fakenews)
    true_df['label']=1
    fake_df['label']=0

    #Join dataframes
    df=pd.concat([true_df, fake_df])
    print("Done")
    
    #Split
    _, train_df, test_df=train_test_split(df, training_data_fraction=training_data_fraction, class_column_name='label', return_numpy=False)
    
    Y_train=train_df['label'].to_numpy()
    Y_test=test_df['label'].to_numpy()

    Y_train_neg=Y_train.copy()
    Y_train_neg[Y_train_neg==0]=-1
    Y_test_neg=Y_test.copy()
    Y_test_neg[Y_test_neg==0]=-1

    Y_train_enc=labels_encoding(Y_train, labels=[0,1])
    Y_test_enc=labels_encoding(Y_test, labels=[0,1])

    Y_TRAIN={'LogisticRegression': Y_train, 'LinearSVM': Y_train_neg, 'MLP': Y_train_enc}
    Y_TEST={'LogisticRegression': Y_test, 'LinearSVM': Y_test_neg, 'MLP': Y_test_enc}
    
    return Y_TRAIN, Y_TEST, train_df, test_df
    
    







def run_experiments(
    column_order, train_df, test_df, Y_TRAIN, Y_TEST,
    z_score_options=[],
    sparse_options=[],
    columns_list=[],
    models=[],
    C_list=[],
    hidden_layer_widths_list=[],
    vectorizations=[],
    tokenizers=[],
    stop_words_options=[],
    ngram_ranges=[],
    max_features_list=[],
    activations=[],
    lrs=[],
    bss=[],
    epochs_list=[],
    compute_metrics=False,
    metrics_dict=None,
    filename=None
):
    try:
        results=[]
        result_df=pd.DataFrame({})
        METRICS_HISTORIES=[]
        
        def display_results():
            df_result=pd.DataFrame(results, columns=column_order)
            clear_output(wait=True)
            display(df_result.style.hide(axis="index"))

        vect_configs=[{'vectorization': vectorization,
                        'tokenizer': tok_name,
                        'stop_words': stop_words,
                        'ngram range': ngrams,
                        'max features': max_features,
                        'z-score': z_score, 
                        'columns': columns, 
                        'sparse': sparse}
                    for sparse in sparse_options for z_score in z_score_options for columns in columns_list for max_features in max_features_list
                    for ngrams in ngram_ranges for vectorization in vectorizations for stop_words in stop_words_options
                    for tok_name in tokenizers.keys() if not (max_features is None and not sparse)] 
                    #non sparse computation takes to much storage if max_features None (i.e. unbounded)
        

        for vcf in vect_configs:
            #Vectorize:
            vect_start=time.time()
            vectorizer=multi_column_vectorizer(col_names=vcf['columns'], vectorization=vcf['vectorization'], max_features_per_column=vcf['max features'],
                                            ngram_range=vcf['ngram range'], stop_words=vcf['stop_words'], tokenizer=tokenizers[vcf['tokenizer']])
            X_train=vectorizer.fit_transform(train_df, sparse=vcf['sparse'])
            X_test=vectorizer.transform(test_df, sparse=vcf['sparse'])
            
            length=X_train.shape[0] #length of X_train
            
            if vcf['z-score']:
                #Apply z-score normalization
                if vcf['sparse']:
                    X=sp.vstack([X_train, X_test])
                    X, offset=standardize_sparse_matrix(X)
                    X_train, X_test=X[:length], X[length:]
                else:
                    X=np.vstack([X_train, X_test])
                    X=standardize(X)
                    X_train, X_test=X[:length], X[length:]
                    offset=None
            else:
                offset=None
                    
            vect_end=time.time()
            
            num_features=X_train.shape[1]
            
            models_configs=[]
            models_configs+=[{'model':'LogisticRegression',
                            'activation': None,
                            'C': None,
                            'widths':None, 
                            '# epochs': epochs,
                            'learning rate': lr,
                            'batch size':bs}
                            for m in models for epochs in epochs_list for bs in bss
                            for lr in lrs if m=='LogisticRegression'] #Add LinearRegression models
            
            models_configs+=[{'model':'LinearSVM',
                            'activation': None,
                            'C': C,
                            'widths':None, 
                            '# epochs': epochs,
                            'learning rate': lr,
                            'batch size':bs}
                            for m in models for epochs in epochs_list 
                            for bs in bss for lr in lrs for C in C_list if m=='LinearSVM'] #Add LinearSVM models
            
            models_configs+=[{'model':'MLP',
                            'activation': activation,
                            'C': None,
                            'widths':[num_features]+hidden_layer_widths+[2], 
                            '# epochs': epochs,
                            'learning rate': lr,
                            'batch size':bs}
                            for m in models for epochs in epochs_list for lr in lrs for hidden_layer_widths in hidden_layer_widths_list
                            for bs in bss for activation in activations   #Add MLP models
                            if m=='MLP'if not vcf['sparse']]  # our MLP models don't support sparse matrix computations)

            #Initialize optimizers
            optimizers={lr: GDOptimizer(lr) for lr in lrs}

            for mcf in models_configs:
                modelname=mcf['model']
                y_train=Y_TRAIN[modelname]
                y_test=Y_TEST[modelname]
                #Initialize model
                if modelname=='LogisticRegression':
                    model=LogisticRegression(w=np.zeros(num_features), b=0, optimizer=optimizers[mcf['learning rate']], offset=offset)
                elif modelname=='LinearSVM':
                    model=LinearSVM(w=np.zeros(num_features), b=0, optimizer=optimizers[mcf['learning rate']], C=mcf['C'], offset=offset)
                elif modelname=='MLP':
                    model=MLP(widths=mcf['widths'], optimizer=optimizers[mcf['learning rate']], activation=mcf['activation'])
                else:
                    raise Exception(f'Unsupported model: {modelname}')

                #Train model
                train_start=time.time()
                metrics_dict_=metrics_dict
                if not metrics_dict is None and'accuracy' in metrics_dict:
                    metrics_dict_['accuracy']=ACCURACIES[modelname]
                metrics_history=model.fit(X_train, y=y_train, num_epochs=mcf['# epochs'], batch_size=mcf['batch size'],
                                           compute_metrics=compute_metrics, metrics_dict=metrics_dict_)
                train_end=time.time()
                #Evaluate
                train_accuracy=np.round(ACCURACIES[modelname](y_pred=model.decision_function(X_train), y_true=y_train),4)
                test_accuracy=np.round(ACCURACIES[modelname](y_pred=model.decision_function(X_test), y_true=y_test),4)

                #Save and display results

                result={'# features': num_features,
                        'vectorization time [s]': np.round(vect_end-vect_start,2),
                        'train accuracy [%]': train_accuracy,
                        'test accuracy [%]': test_accuracy,
                        'training time [s]': np.round(train_end-train_start,2)
                        }
                result.update(vcf) #Save vectorization configs
                result.update(mcf) #Save model configs
                results.append(result)
                METRICS_HISTORIES.append((result, metrics_history))
                display_results()

        result_df=pd.DataFrame(results)
        return result_df, METRICS_HISTORIES
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        if filename is not  None:
            result_df.to_csv(filename)
        
    

             

    
    
        
    

    
                                                    
                                            

                                            
                        
                        
                        
                
        
        
        