# 🚲 Citibike Stations: Prediction of Bike/Dock Availability

In this project, we will use **historic station status data** (docks available, bikes available @ timestamp) to predict the bike/dock availability at another future timestamp.

![](images/citi_bike.webp)
## Instructions to Run

To run the notebook and deployment of the project:
1. Clone the repository:
    ```
    git clone https://github.com/datasciencedani/citibike-stations-ml.git
    ```

1. Ensure environment with `pipenv`:
    ```
    pip install pipenv
    ```
    ```
    pipenv install -r requirements.txt
    ```
    ```
    pipenv shell
    ```
1. Create a jupyter kernell for the environment:
    ```
    python -m ipykernel install --user --name=env-ml-citibike
    ```

## Notebooks

> To run this notebooks select the `env-ml-citibike` kernel in your jupyter lab notebook.

1. [Data Cleaning](nbs/00_data_cleaning.ipynb): notebook where we talk about the data we will be using ([Kaggle Citi Bike Stations](https://www.kaggle.com/datasets/rosenthal/citi-bike-stations/)) and perform several steps to clean the dataset and create the label we will use for prediction (`bike/dock percentage availability`).
2. [Feature Engineering & EDA](nbs/01_feature_eng.ipynb): notebook where we perform an exploratory analysis on our dataset (availability over stations and over different days/times of the day), and prepare our features for future modeling.
3. [Training](nbs/02_training.ipynb): notebook where we perform model training and hyperparameter tuning to find the best tree-based model that accommodates our data 
    > Disclaimer🚨: the resulting model does not perform as expected, but serves as a proof of concept of what can be done. We expect to improve the performance of our predictions by trying different architectures (deep learning - next topic) and also by improving our features (adding weather data and transformations to our time variables).

## Model Deployment

1. Build the docker container.
    ```
    docker build -t citibike-test .
    ```
1. Run our container exposing the prediction port:
    ```
    docker run -it --rm -p 8080:8080 citibike-test
    ```
1. Run code in `predict_test.ipynb`.