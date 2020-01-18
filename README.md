# Fake-News-Detection
With the development of social media, such as Twitter, Facebook, Weibo and so on, there are more available outlets for the society to obtain the current affairs. Differing from the traditional media platforms, the spread of news in these new social media is much quicker and easier. At the same time, there are more and more fake news in these media platforms. A most significant problem is that some false news that are eye-catching to increase the number of users to a website are produced.
In order to decrease such instances and rate various websites for being trustworthy, we have an idea to build the model that can identify the relationship between headline and article.
In this project, we created a model that finds the similarity between the Headline and the article Body and classifies it into 4 classes: unrelated, discuss, agree and disagree. The model uses Bi-directional LSTM for summarizing the sequence and CNN for classifying the summarized sequence.

Explaination to these files:

4 csv.file:
train_bodies.csv & train_stances.csv: training data from the website:fakenewschallenge.org

body.csv & headline.csv: sample data for our designed detector (for more detail, please the part G in our report)

4 py.file
makingEmbedding.py: used to read the data and convert the words into word vectors
classification_gridsearch2.py: used to build the model, and find the optimized parameters by doing grid searching
dd.py: after obtaining the optimized model, used it to check the accuracy of our model (including plot of accuracy & loss with epochs, confusion matrix for testing data, and so on)
lkl.py: used as a detector to identify the relationship between the input headline and body article

glove.6B.50d.txt: the GloVe (word vector) with 50 dimensions downloaded from https://nlp.stanford.edu/projects/glove/

the restdocuments, incluing the .txt,.p,.png, actually are the production of our model.
