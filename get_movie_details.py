
# coding: utf-8

# In[1]:


import pandas as pd
import json, requests
from IPython.display import Image
from IPython.display import display
import urllib


# In[2]:


def read_data(filepath = 'ml-100k/u.item'):
    movie_titles = []
    with open(filepath, mode = 'r') as f:
        for line in f.readlines():
            movie_titles.append(line.split('|')[1])
    return movie_titles


# In[6]:


def get_movie_details(movie_titles):
    movie_details = []
    headers = {'Accept': 'application/json'}
    payload = {'api_key': '85f9170ef8852f18ef6500d621c5c7fc'}  
    for i, movie_title in enumerate(movie_titles):
        payload['query'] = movie_title.split('(')[0] 
        try:
            response = requests.get('http://api.themoviedb.org/3/search/movie/', params = payload, headers = headers)    
            movie_details.append(json.loads(response.text)['results'][0])
        except:
            movie_details.append()
    return pd.DataFrame(movie_details)


# In[5]:


def download_posters():
    headers = {'Accept': 'application/json'}
    payload = {'api_key': '85f9170ef8852f18ef6500d621c5c7fc'} 
    response = requests.get("http://api.themoviedb.org/3/configuration", params=payload, headers=headers)
    response = json.loads(response.text)
    base_url = response['images']['base_url'] + 'w185'
    for i, movie in enumerate(movie_details):
        image_url = base_url + movie['poster_path']
        with open('images/{}.jpg'.format(i), 'wb') as f:
            resource = urllib.urlopen(image_url)
            f.write(resource.read())
            f.close()

