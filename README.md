# Fetal Health Classifier
Building a Gradient Boosting Classifier to detect pathological samples from fetal health data.
## Fetal Health Classifier - Python
Python Jupyter Notebook that loads in raw tabular data containing features extracted from Cardiotocogram exams. The notebook leverages advanced ensemble learning techniques like boosting to train several models, evaluating the performance of various boosting algorithms in classifying fetal health. The best model is chosen for further hyperparameter tuning, where the optimal model is found.
### Data Resource - Kaggle
This dataset was sourced from Kaggle. Click [here](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) for access to the dataset.
### Method
The data is first loaded using the pandas library. The data is preprocessed and assessed for building the model. After processing, the data is split into test and train subsets to properly evaluate model performance. Moreover, because the dataset features a class imbalance, model training will be done using stratified sampling of the data. The next step is to create a baseline weak learner model. This is so that we can assess the effectiveness of advanced classification techniques against traditional models. Using Scikit-Learn, a basic decision tree is trained to evaluate the performance of one weak learner. Next, train several boosting algorithms to find the best base model to tune. In this project, I evaluate the following algorithms: AdaBoostClassifier, GradientBoostingClassifier, and HistGradientBoostingClassifier. The models are evaluated using F1-Score because the dataset is imbalanced. The best performing model combination is further tuned by finding the best hyperparameters via Scikit-Learn's random search. In this test environment, HistGradientBoostingClassifer was most optimal, and another library, scipy, was simply used to create parameter distributions for certain hyperparameters. Finally, the best model is evaluated for its performance, comparing its classification report and confusion matrix against its baseline model, and both the base model and tuned model paramters are displayed.

## Code Dependencies, Pre-requisites, and Version History
### Dependencies
The program requires the following libraries:
1) Pandas
2) Matplotlib
3) Seaborn
4) Scikit-Learn
5) Scipy

The notebook was tested using Python version 3.13.9.

### Pre-Requisites and Setup
To install missing external libraries on your local machine, open the command prompt and use the following command:

    pip install <library_name>

For the notebook to run properly, ensure the following files and directories exist in the same directory:
1) fetal_health.csv
2) Fetal_Health_Classifier_Model.ipynb

### Version History
V1.0 - The Jupyter Notebook is created. All cells and functions have been tested and are functional. Optimal model is found.
V1.1 - Added F1-Score metric print-outs to facilitate model evaluation.

## Run Instructions
Once the dependencies are installed and the pre-requisites and setup have been completed, you are all set to run the notebook.
### Instructions
1) Open IDE.
2) Open the directory containing the notebook and the dataset directory.

Using the notebook:
1) Open the notebook in the IDE.
2) Run all cells in the notebook.
3) Validate model training results using the displayed evaluation metrics, including the classification report and confusion matrix. In production, I obtained a test F1-Score of 94%. Results should be similar to this. Furthermore, despite similar F1-Scores to the base model, the biggest observation should be that the tuned model accurately classifies all pathological samples, which is a critical aspect in the application of this model.
4) Best model can now be used in other applications or further tuned.



## Fetal Health Optimization - Python
Python Jupyter Notebook that benchmarks the baseline model against an optimized version of the model. The notebook leverages advanced optimization techniques like efficient numerics representations and model complexity reduction, evaluating the performance and efficiency across both models in classifying fetal health. The results of model optimization are discussed and evaluated.
### Method
The baseline model code is copied and pasted from the original model notebook. By setting random state seeds in the original model's production, copying and pasting will produce the same model. The pandas library is used to benchmark memory usage, the time library is used to benchmark training time and prediction speed, and scikit-learn, matplotlib, and seaborn are used to benchmark model performance. 
After benchmarking the baseline model, an optimized model is built. Efficient numerics are implemented by changing the data type of the feature data. Model complexity is reduced by adding new hyperparameters and modifying existing hyperparameters to reduce training time. The benchmarks are then measured the same way. Finally, benchmark metrics between the baseline and optimized model are compared to assess the impacts of optimization. 

## Code Dependencies, Pre-requisites, and Version History
### Dependencies
The program requires the following libraries:
1) Pandas
2) Matplotlib
3) Seaborn
4) Scikit-Learn

The notebook was tested using Python version 3.13.9.

### Pre-Requisites and Setup
To install missing external libraries on your local machine, open the command prompt and use the following command:

    pip install <library_name>

For the notebook to run properly, ensure the following files and directories exist in the same directory:
1) fetal_health.csv
2) Fetal_Health_Classifier_Optimization.ipynb

### Version History
V1.0 - The Jupyter Notebook is created. All cells and functions have been tested and are functional. Optimized model is developed.
V1.1 - Update markdown cells to have more accurate information.

## Run Instructions
Once the dependencies are installed and the pre-requisites and setup have been completed, you are all set to run the notebook.
### Instructions
1) Open IDE.
2) Open the directory containing the notebook and the dataset directory.

Using the notebook:
1) Open the notebook in the IDE.
2) Run all cells in the notebook.
3) Validate model training results using the displayed evaluation metrics. In production, I observed significantly reduced training times, prediction speeds, and memory usage. Furthermore, prediction performance has also improved, increasing true suspect labels which is critical in this domain. 
4) Optimized model can now be used in other applications or further tuned.
