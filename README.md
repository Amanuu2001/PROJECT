import requests
from bs4 import BeautifulSoup

#An example showing the scrapping of reviews of Ethiopian airlines from the airlinequelity.com website

url = 'https://www.airlinequality.com/airline-reviews/ethiopian-airlines/page/5/?sortby=post_date%3ADesc&pagesize=100'

r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
header_tags = soup.find_all('h2', {'class':'text_header'})
content_tags = soup.find_all('div', {'class':'text_content'})
rating_tags = soup.find_all('span', {'itemprop':'ratingValue'})
headers = [x.text for x in header_tags][:100]
content = [x.text for x in content_tags]
ratings = [x.text for x in rating_tags][1:]
print(len(headers), len(content), len(ratings))

#This function allows us to modify the url in each iteration so that we get the desired urls
def url_cycler(url_slice1, url_slice2):
  for i in range(7):
    yield url_slice1 + '{}'.format(i+1) + url_slice2

#Used later to extract the relevant review string
def strip(x):
  return x.text.split('\n')[1].split(' ')[-1].split('\r')[0]

#Function used to identify desired class for the type of review
def review_rating_header(tag):
  if 'class' in tag.attrs.keys():
    return True if 'review-rating-header' in tag['class'] else False
  else:
    return False

#Function used to identify desired class for the type of review
def review_value_or_stars(tag):
  #print(tag.attrs.keys())
  if 'class' in tag.attrs.keys():
    #print(tag['class'])
    #tag['class'] returns a list of classes the tag belongs to
    return True if ('review-value' in tag['class'] or 'review-rating-stars' in tag['class']) else False
  else:
    return False

#First half of the url
url1 = 'https://www.airlinequality.com/airline-reviews/ethiopian-airlines/page/'
#Second half of the url
url2 = '/?sortby=post_date%3ADesc&pagesize=100'

#Instantiate the generator that combines the two urls in a way that a different url with the desired name is generated each time
url_iter = url_cycler(url1, url2)

#A list containing the titles of the reviews
titles = []
#A list containing the body of the reviews
content = []
#A list containing the numerical reviews out of 10
ratings = []
#Different types of ratings
various_ratings = []
#categories set
category_types = set()
#Iterating over each url
for i, url in enumerate(url_iter):
  #if i != 6:
    #continue
  #get the url
  r = requests.get(url)
  #Soupify the the html
  soup = BeautifulSoup(r.text, 'html.parser')

  #Find all 'h2' tags with class: 'text_header'
  #After inspecting the website I determined the tags containing the titles of the reviews are h2 tags with class name: 'text_header'
  title_tags = soup.find_all('h2', {'class':'text_header'})
  #Find all 'div' tags with class name: 'text_content'
  content_tags = soup.find_all('div', {'class':'text_content'})
  #Find all 'div' tags with class name: 'rating-10'
  rating_tags = soup.find_all('div', {'class', 'rating-10'})
  #Find all 'table' tags with class name: 'review-ratings'
  review_details = soup.find_all('table', {'class':'review-ratings'})

  #Append each list with the correspondin items
  #Some of the ratings need to be stripped of \n, \r, and spaces (function shown above)
  ratings = ratings + [None if strip(x) == 'na' else int(strip(x)[0])/2 for x in rating_tags if strip(x) != '']
  content = content + [x.text for x in content_tags]

  #The last 4 titles of the 7th(last) page are not review titles
  if i == 6:
    titles = titles + [x.text for x in title_tags][:-4]
  #The first 100 titles are review titles
  else:
    titles = titles + [x.text for x in title_tags][:100]
  #print(review_details[1].prettify())

  #Iterate over each rating block of all reviews in page
  for j, rating_block in enumerate(review_details):
    if j == 0:
      continue
    categories = []
    results = []
    categories_and_results_dict = {}
    #For each category find the name of the category and append to the categories list
    #print(rating_block.prettify())
    for category in rating_block.find_all(review_rating_header):
      categories.append(category.text)
      category_types.add(category.text)
    #For each result corresponding to each category append to results list
    for result in rating_block.find_all(review_value_or_stars):
      #if review type is letters
      if 'review-value' in result['class']:
        results.append(result.text)
      #if it is an out of 5 rating
      else:
        #The number of stars given is obtained by counting the tags with class name 'star fill'
        results.append(len(result.find_all('span', {'class':'star fill'})))
    for k in range(len(categories)):
      categories_and_results_dict[categories[k]] = results[k]
    various_ratings.append(categories_and_results_dict)
    #Then zip each element from categories with the corresponding element in results
    #print(categories)
    #print(results)

    #ratings_zip = zip(categories, results)
    #various_ratings.append(ratings_zip)

category_types_list = list(category_types)

#Give indeces to various_ratings
for i, item in enumerate(various_ratings):
  item['index'] = i

#Creating a dataframe to store the data
import pandas as pd
import numpy as np
ind = [i for i in range(len(titles))]
flights_df1 = pd.DataFrame({'index':ind, 'title':titles, 'review_content':content, 'overall_rating':ratings})
flights_df2 = pd.DataFrame(various_ratings)
flights_df = flights_df1.merge(flights_df2, on='index')
flights_df = flights_df.fillna(value=np.nan)

#We substract 3 from the reviews to set the more negative reviews to be less than 0 and the more positive reviews more than 0
flights_df['Seat Comfort'] = flights_df['Seat Comfort'].fillna(3) - 3
flights_df['Cabin Staff Service'] = flights_df['Cabin Staff Service'].fillna(3) - 3
flights_df['Food & Beverages'] = flights_df['Food & Beverages'].fillna(3) - 3
flights_df['Ground Service'] = flights_df['Ground Service'].fillna(3) - 3
flights_df['Wifi & Connectivity'] = flights_df['Wifi & Connectivity'].fillna(3) - 3
flights_df['Inflight Entertainment'] = flights_df['Inflight Entertainment'].fillna(3) - 3

print(flights_df.head())
print(flights_df.shape)
#print(flights_df.info())

#Save the reviews and the ratings
flights_df.to_csv('flights_data.csv', index=False)

#Preparing the Dataset for TfIdf vectorizer
from sklearn.model_selection import train_test_split

flights_df = pd.read_csv('flights_data.csv')

#The X value here is the review content text and the y values are the 'Recommended' columns which will be used for fitting for NMF model
X_train, X_test, y_train, y_test = train_test_split(flights_df[['review_content', 'title']], flights_df[['Recommended', 'Seat Comfort', 'Cabin Staff Service', 'Food & Beverages', 'Ground Service', 'Wifi & Connectivity', 'Inflight Entertainment']], test_size=0.2, random_state=53)

from sklearn.feature_extraction.text import TfidfVectorizer
#Initializing the TfIdf Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
#Training the train data
tfidf_train = tfidf_vectorizer.fit_transform(X_train['review_content'])
#Training the test data
tfidf_test = tfidf_vectorizer.transform(X_test['review_content'])
#The split train and test tfidf vectors are used later for other models while the non-split tfidf vectors are used in a different model for logistic regression and are further split then

#Training the total data(without spliting to train and test); used for later
tfidf_total = tfidf_vectorizer.transform(flights_df['review_content'])
print(tfidf_train.A.shape)
#print(tfidf_vectorizer.get_feature_names_out())
print(len(tfidf_vectorizer.get_feature_names_out()))
#print(tfidf_train.A[100][np.argmax(tfidf_train.A[100])])
print(tfidf_vectorizer.get_feature_names_out()[441])

print(tfidf_total.A.shape)

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

model = NMF(n_components = 15, max_iter=5000)
nmf_features_total = model.fit_transform(tfidf_total.A)
norm_features_total = normalize(nmf_features_total)
components_total = model.components_
print(norm_features_total.shape)
print(components_total.shape)

#Save the trained data to a dictionary
NMF_features_train_dict = {}
for i in range(norm_features_train.shape[1]):
  a = 'Review - {}'.format(i+1)
  NMF_features_train_dict[a] = list(norm_features_train[i])

#Listing the most important words obtained using the NMF model
array_total = []
for i in range(len(components_total)):
  array_total.append([tfidf_vectorizer.get_feature_names_out()[np.argmax(components_total[i])], components_total[i, np.argmax(components_total[i])]])
  #print(tfidf_vectorizer.get_feature_names_out()[np.argmax(components_train[i])],'-', components_train[i, np.argmax(components_train[i])])
index_list = [i for i in range(len(array_total))]
np_array_total = np.array(array_total)
words_df_total = pd.DataFrame({'index':index_list, 'word':np_array_total[:,0]})
print(words_df_total)

#Listing the most important words obtained using the NMF model
array_total = []
for i in range(len(components_total)):
  array_total.append([tfidf_vectorizer.get_feature_names_out()[np.argmax(components_total[i])], components_total[i, np.argmax(components_total[i])]])
  #print(tfidf_vectorizer.get_feature_names_out()[np.argmax(components_train[i])],'-', components_train[i, np.argmax(components_train[i])])
index_list = [i for i in range(len(array_total))]
np_array_total = np.array(array_total)
words_df_total = pd.DataFrame({'index':index_list, 'word':np_array_total[:,0]})
print(words_df_total)

#Train and test algorithms for Decision tree and random forest classifier
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

model = NMF(n_components = 15, max_iter=8000)

nmf_features_train = model.fit_transform(tfidf_train.A)
norm_features_train = normalize(nmf_features_train)
components_train = model.components_

nmf_features_test = model.fit_transform(tfidf_test.A)
norm_features_test = normalize(nmf_features_test)
components_test = model.components_

print(norm_features_train.shape)
print(model.components_.shape)

#There are 40 topics

#A different type of algorithm to use on the NMF trained components
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
DT = DecisionTreeClassifier()
DT.fit(NMF_X_train, NMF_y_train)
pred_DT = DT.predict(NMF_X_test)
print(roc_auc_score(pred_DT, NMF_y_test))
print(accuracy_score(pred_DT, NMF_y_test))

#Another type of algorithm
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(max_depth=2, random_state=0)

RF.fit(NMF_X_train, NMF_y_train)
pred_RF = RF.predict(NMF_X_test)
print(roc_auc_score(pred_RF, NMF_y_test))
print(accuracy_score(pred_RF, NMF_y_test))

!pip install dash

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash()

areas_of_improvement = pd.read_csv('Areas_of_Improvement.csv')

flights_df = pd.read_csv('flights_data.csv')

fig1 = px.bar(areas_of_improvement, x='word', y='coefficients')

fig2 = px.histogram(flights_df, x='overall_rating')

app.layout = html.Div(children=[
    html.H1(children='Dashboard'),

    html.Div(children='''
        Areas of Improvement Plot
    '''),

    dcc.Graph(
        id='AOI graph',
        figure=fig1
    ),

    html.Div(children='''
        Overall ratings plot
    '''),

    dcc.Graph(
        id='Ratings count graph',
        figure=fig2
    )
])

if __name__ == '__main__':
    app.run(debug=True)
