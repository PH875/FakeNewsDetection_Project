## Machine Learning Project: Fake News Detection

In this project, we tested different models and text preprocessing methods using the ["fake-and-real-news-dataset"](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). For more information see our report. 


### Setup
1. Clone this repository
   ```bash
   git clone https://github.com/PH875/FakeNewsDetection_Project.git
   ```
   or download it otherwise.
2. In the terminal, navigate to this repository
3. Create the environment. Using conda (recommended) run:
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
5. Run all the code from this environment. To reproduce the results from our report, run this [notebook](fakenews_detection.ipynb)


Notizen: 
Begründung für z-score normalization
Courselib Änderungen: LogisticRegression model, LinearSVM model, TrainableModel, normalization, preprocessing

