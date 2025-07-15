# Machine Learning Project: Fake News Detection

In this project, we tested different models and text preprocessing methods for fake news detection using the [fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). For more information check out our [**report**](report.pdf). 


## Setup
1. Clone this repository
   ```bash
   git clone https://github.com/PH875/FakeNewsDetection_Project.git
   ```
   or download it otherwise.
2. In the terminal, navigate to this repository.
3. Create the environment. Using conda (recommended), run:
   ```bash
   conda env create -f environment.yml
   ```
   If using pip, activate or create your prefered environment and run:
   ```bash
   pip install -r requirements.txt
   ```
   
4. For conda users: Activate the environment:
   ```bash
   conda activate fakenews_detection
   ```
5. Make sure to run all the code from this repository in this environment.


## Running Code
To reproduce the results from our report, run [**this**](fakenews_detection.ipynb) jupyter notebook. 

## Results
When trained on titles only, our best models could achieve 100% train and up to 96% test accuracy. When trained on both title and text of the news articles, we could even achieve test accuracies of around 99%, again with 100% train accuracy. For more detailed results, check out our [**report**](report.pdf). 

Moreover you can find some of our preliminary testing results in [preliminary experiments](preliminary_experiments), where we tested different model configurations to find out which configurations work best for our final experiments. (But note that they are not part of the official experiment.)


## Changes and Extensions to `courselib`

We used the courselib library from the course [AppliedML](https://github.com/mselezniova/AppliedML/tree/main) with the following changes/extensions made:
- added `precision`, `recall` and `f1_score` in [`metrics`](courselib/utils/metrics.py)
- added `lp_normalize` for $Lp$-normalization and `standardize_sparse_matrix` for efficient z-score normalization of sparse matrices in [`normalization`](courselib/utils/normalization.py)
- made [`TrainableModel`](courselib/models/base.py), [`LogisticRegression`](courselib/models/glm.py) and [`LinearSVM`](courselib/models/svm.py) compatible with sparse matrix operations and added "offset-functionality" in [`LogisticRegression`](courselib/models/glm.py) and [`LinearSVM`](courselib/models/svm.py)  to efficiently apply z-score normalization to sparse matrices combined with `standardize_sparse_matrix`
- added `basic_word_tokenizer`, `lemmatization_tokenizer`, `stemming_tokenizer` and `multi_column_vectorizer` in [`preprocessing`](courselib/utils/preprocessing.py)
for text tokenization and vectorization
- added `binary_confusion_matrix` for confusion matrices of data with binary labels and slightly changed `plot_confusion_matrix` to allow for more customization in [`metrics`](courselib/utils/metrics.py)


## LICENSE
This project is licensed under the terms of the [MIT license](LICENSE).


