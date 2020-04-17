# Reddit-flair-classifier
A reactjs app with a flask backend which classifies reddit posts using logistic regression from r/india into 6 flair categories - 
1. 'Coronavirus'
2. 'Science/Technology'
3. 'Policy/Economy'
4. 'Politics'
5. 'Non-Political'
6. 'AskIndia'

## Codebase

The data collection, evaluation and model training was done using Google Colab. The backend is being served using Flask and the frontend was developed using React.js. The backend is hosted using heroku and the frontend is hosted on github pages.

## Respository Structure

1. The [base directory](https://github.com/dh1105/Reddit-flair-classifier) of the repo consists of the Flask API files, the Heroku Procfile, the Reddit Connector class and the Text classification helper class. These are all needed to run the API, load 
the model and to connect with reddit. It also consists of the logistic regression model, the count vectorizer file, the tfidf transformer file, the tokenizer file and a trained word2vec LSTM model.

2. The [src](https://github.com/dh1105/Reddit-flair-classifier/tree/master/src) consists of the reactjs code which is used to design and run the webapp.

3. The [notebooks](https://github.com/dh1105/Reddit-flair-classifier/tree/master/notebooks) folder consists of all the 3 jupyter notebooks used to collect the data, evaluate the data and train the necessary classifiers.

## Running this project

1. Clone this repository using
  ```
  git clone https://github.com/dh1105/Reddit-flair-classifier.git
  cd Reddit-flair-classifier
  ```

2. Ensure that Python 3.x is installed on your local system. Install all dependencies using 
  ```
  pip install -r requirements.txt
  ```

3. In order to run the Flask API, you need to have a Reddit account with an api key to access data. Modify the [RedditConnector.py](https://github.com/dh1105/Reddit-flair-classifier/blob/master/RedditConnector.py) class and add your details to it.

4. You can now run the Flask API using ```python app.py```. This API will be listening at http://localhost:5000/. You can send requests to the respective endpoints.
   - A sample call to the '/predict' endpoint would be:
   ```
   POST /predict HTTP/1.1
   Host: localhost:5000
   Content-Type: application/json
   cache-control: no-cache
   Postman-Token: 685cb700-05d5-49cd-8663-bb3d83abb4c8
   {
	  "url": "https://www.reddit.com/r/india/comments/g1v3cn/what_are_you_watching/?utm_source=share&utm_medium=web2x"
   }------WebKitFormBoundary7MA4YWxkTrZu0gW--
   ```
   
   - A sample call to the '/automated_testing' endpoint would be:
   ```
   POST /automated_testing HTTP/1.1
   Host: localhost:5000
   Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW
   cache-control: no-cache
   Postman-Token: 87036eae-7239-41e4-a045-c6e8d77a1ae5

   Content-Disposition: form-data; name="file"; filename="C:\Reddit_flair_flask_app\example.txt


   ------WebKitFormBoundary7MA4YWxkTrZu0gW--
   ```
   Take note that this expects a file with a link to a Reddit post on every line. The file can have any name but the form data **key must be 'file'**.

5. To start the frontend, ensure that you have npm installed. You can start the app by issuing the following command.
  ```
  npm start
  ```

6. The backend and frontend can be run independently as well. By default, the front-end will be making API calls to the Heroku backend. In order to allow calls to be made to the local Flask app, uncomment line 1 and comment line 2 in [baseURL.js](https://github.com/dh1105/Reddit-flair-classifier/blob/master/src/baseURL.js).

## Dataset 

The dataset consists of 36000 posts, 6000 from each flair, all stored in a .csv file. It is available for download [here](https://drive.google.com/open?id=1wq1ETFh0P61zdDFwlZ6583b-H20vBWar).

## Methodology

The main logic behind the project is in the three jupyter notebooks. 

1. [Reddit_data_scraper](https://github.com/dh1105/Reddit-flair-classifier/blob/master/notebooks/Reddit_data_scraper.ipynb): Using the pushshift.io API, I downloaded some of the latest posts from Reddit r/india. Each flair I considered has 6000 posts in the dataset.
In addition to this, it also contains some baseline traditional ML models trained on the dataset. The data collected consisted of a variety of fields but only the 'Title', 'Selftext' and 'link_flair_text' were used further in the problem.

2. [Exploratory_data_analysis](https://github.com/dh1105/Reddit-flair-classifier/blob/master/notebooks/Exploratory_data_analysis.ipynb): Analysis of the dataset to find frequent words corresponding to each flair, see the distribution of invalid text in posts [NaN, [deleted], [removed]] and to find prominent features to be used as training data.
   - An evaluation of the 'selftext' revealed that a majority of posts either did not contain any text or had the text removed. As a result, the body itself would not serve as a suitable feature. Therefore, the 'title' and 'selftext', if any, were combined to make the feature to be considered.
   - This data was cleaned by removing punctuations, stopwords and URLs. The cleaned data was tokenized and displayed in seperate word clouds for each flair. This gave an insight about the significant overlap of key words between classes.
   - Four models were trained as baseline models using a combination of the 'title' and 'selftext' as a feature. The Logistic regression model performed the best out of these four.

3. [Flair_classification](https://github.com/dh1105/Reddit-flair-classifier/blob/master/notebooks/Flair_classification.ipynb): An attempt to enhance the traditional ML algorithms by using LSTMs to try and classify posts. The evaluation was done using simple LSTMs with both pre-trained embedding layers and trainable embedding layers.
   - Data cleaning was the same as mentioned earlier and the input consisted of the 'title' and 'selftext', if any, concatenated. Four embedding types were considered for the LSTM embedding layer - pre-trained word2vec, pre-trained Fasttext, pre-trained GLoVe and a trainable embedding layer. 
   - Four models were trained with each respective embedding layer. Each model also had EarlyStopping as a callback to prevent overfitting.
   - The word2vec model outperformed the other three models. However, I was unable to use that with the Flask API due to Heroku's slug size constraints. 
   
## Results

### Traditional ML models

| Model                                 | Testing accuracy | Training accuracy |
| ------------------------------------- | ---------------- | ----------------- |
| Logistic regression                   | 0.5944           | 0.7848            |
| Support Vector Machine                | 0.5971           | 0.9352            |
| Multinomial Naive Bayes               | 0.5833           | 0.7348            |
| Stocastic Gradient Descent classifier | 0.5758           | 0.7306            |

As evident from the table above, although the SVM does have a marginaly high accuracy than logistic regression, it seems to be overfitting.
A more detailed analysis of the models comprising of the classification report and confusion matrix can be seen in the [Exploratory_data_analysis](https://github.com/dh1105/Reddit-flair-classifier/blob/master/notebooks/Exploratory_data_analysis.ipynb) notebook.

### Deep learning models

| Model                                 | Testing accuracy and loss | Validation accuracy and loss in the final epoch |
| ------------------------------------- | ------------------------- | ----------------------------------------------- |
| Word2Vec LSTM                         | 0.521, 1.273              | 0.5177, 1.2955                                  |
| GloVe LSTM                            | 0.382, 1.576              | 0.3729, 1.5764                                  |
| Fasttext LSTM                         | 0.518, 1.282              | 0.5177, 1.3147                                  |
| Trainable embedding LSTM              | 0.551, 1.552              | 0.5625, 1.5041                                  |

As evident from the table above, the GLoVe did not perform well. The model with the trainable embedding layer seems to be performing the best in terms of accuracy. However, it started overfitting very early and 
has one of the highest losses. You can see the plot of epoch vs acc and loss in the [Flair_classification](https://github.com/dh1105/Reddit-flair-classifier/blob/master/notebooks/Flair_classification.ipynb) notebook.

## Inference

The logistic regression model with the 'title' and 'selftext' combined as a feature seems to be performing the best on the dataset which I had made. It seems to outperform the DL models due to scarcity of data. Hence, as a result 
this was the model which is being used as part of the API. I also wanted to deploy the word2vec LSTM with the Flask API but could not do so due to slug size constraints of Heroku.

## References

1. https://pushshift.io/api-parameters/
2. https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
3. https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
4. https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge
5. https://medium.com/the-andela-way/deploying-a-python-flask-app-to-heroku-41250bda27d0
6. https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
