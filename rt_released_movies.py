"""
Scrapes Rotten Tomatoes for all released (DVD & Streaming) movies (https://www.rottentomatoes.com/browse/dvd-streaming-all/). 
Saves a dataset of reviews from all 'top critics' for each movie.

"""

import time
import datetime

print()
print('Started Loading Movies at:', datetime.datetime.now().strftime('%H:%M:%S'))

import pandas as pd
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from urllib.request import urlopen


# Load all of the released movies and save the page-
browser = webdriver.Chrome(ChromeDriverManager().install())
wait = WebDriverWait(browser, 30)
browser.get('https://www.rottentomatoes.com/browse/dvd-streaming-all/')

while True:
    try:
        more_button = wait.until(EC.visibility_of_element_located((By.CLASS_NAME, 'mb-load-btn'))).click()
    except TimeoutException:
        break
        
content = browser.page_source.encode('utf-8').strip()
all_movies = BeautifulSoup(content, 'html.parser')


print()
print('All Movies Loaded at:', datetime.datetime.now().strftime('%H:%M:%S'))
print()



# Load movie's page and top critic reviews, extract info:
start = datetime.datetime.now()
print('Info Extraction Start Time:', start.strftime('%H:%M:%S'))
print()
all_released_movies = all_movies.find_all('div', attrs={'class': 'movie_info'})

all_top_critic_reviews = []
for i, movie in enumerate(all_released_movies):
    try:
        browser.get('https://www.rottentomatoes.com' + movie.a.get('href'))
        wait.until(EC.visibility_of_element_located((By.ID, 'movie-image-section')))
        content = browser.page_source.encode('utf-8').strip()
        movie_page = BeautifulSoup(content, 'html.parser')

        
        top_critics_link = movie_page.find('p', attrs={'id': 'criticHeaders'}).find_all('a')[1]
        top_critics_url = urlopen('https://www.rottentomatoes.com' + top_critics_link.get('href'))
        top_critics_page = BeautifulSoup(top_critics_url, 'html.parser')
        
        for review in top_critics_page.find_all('div', attrs={'class': 'review_table_row'}):
            row = {}
            row['movie'] = movie.find('h3', attrs={'class': 'movieTitle'}).text
            row['total_reviews'] = int(movie_page.find('div', attrs={'id': 'scoreStats'}).find('span', string='Reviews Counted: ').find_next('span').text)
            row['num_fresh'] = int(movie_page.find('div', attrs={'id': 'scoreStats'}).find('span', string='Fresh: ').find_next('span').text)
            row['num_rotten'] = int(movie_page.find('div', attrs={'id': 'scoreStats'}).find('span', string='Rotten: ').find_next('span').text)
            row['top_critic'] = review.find('div', attrs={'class': 'critic_name'}).find_all('a')[0].text
            row['source'] = review.find('div', attrs={'class': 'critic_name'}).find_all('a')[1].text
            row['freshness'] = review.find('div', attrs={'class': 'review_container'}).div.attrs['class'][-1]
            row['review_date'] = review.find('div', attrs={'class': 'review_date'}).text
            row['review_quote'] = review.find('div', attrs={'class': 'the_review'}).text
            
            all_top_critic_reviews.append(row)
            
        print('Info for movie {} of {} extracted.'.format(i+1, len(all_released_movies))) 
        if ((i+1)%30 == 0):
            print()
            current = datetime.datetime.now()
            lapsed_time = current-start
            avg_per_movie = lapsed_time/(i+1)
            remaining_time_est = avg_per_movie * (len(all_released_movies)-(i+1))
            print('Current Time:', (current).strftime('%m/%d %H:%M:%S'))
            print('Estimated Completion Time:', (current + remaining_time_est).strftime('%m/%d %H:%M:%S'))
            print('Number of Reviews Collected:', len(all_top_critic_reviews))
            print()
    except:
        continue

browser.quit()



# Transform review list to dataframe, calculate tomatometer, save as CSV file:
df = pd.DataFrame(all_top_critic_reviews)
df['tomatometer'] = df.num_fresh / df.total_reviews
columns = ['movie', 'total_reviews', 'num_fresh', 'num_rotten', 'tomatometer', 'top_critic', 'source', 'review_quote', 'freshness', 'review_date']
df = df[columns]
df.to_csv('data/rt_reviews_released_movies.csv', index=False)

