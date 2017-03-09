from __future__ import division
import os
import feedparser
import pocket
import time
import pandas as pd
from pytz import timezone
from watson_developer_cloud import AlchemyLanguageV1
import pickle
from datetime import datetime

feednames = ['theawl.', 'thehairpin.', 'thebillfold.', 'psmag.', 'polygon.', 'arstechnica.', 'politico.', 'fivethirtyeight.',
            'nytimes.', 'thedailybeast', 'citylab.', 'newyorker.', 'motherboard.', 'atlasobscura.', 'digiday.', 'buzzfeed.',
            'longreads.', 'newrepublic.', 'theatlantic.', 'niemanlab.']

#Get URLS for articles
def get_urls():
    urls = ['http://www.theatlantic.com/feed/all/',
    'https://theawl.com/feed',
    'https://thehairpin.com/feed',
    'https://thebillfold.com/feed',
    'https://psmag.com/feed',
    'http://www.polygon.com/rss/index.xml',
    'http://feeds.arstechnica.com/arstechnica/index',
    'http://www.politico.com/rss/politicopicks.xml',
    'https://fivethirtyeight.com/all/feed',
    'http://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'http://feeds.feedburner.com/thedailybeast/articles',
    'http://www.citylab.com/feeds/posts/',
    'http://www.newyorker.com/feed/everything',
    'https://motherboard.vice.com/rss?trk_source=motherboard',
    'http://www.atlasobscura.com/feeds/latest',
    'http://www.niemanlab.org/feed/',
    'http://digiday.com/feed/',
    'https://www.buzzfeed.com/tech.xml',
    'https://www.buzzfeed.com/reader.xml',
    'https://longreads.com/feed/',
    'https://newrepublic.com/rss.xml']
    parsed = [feedparser.parse(l).entries for l in urls]
    links = []
    dates = []
    #Links
    for i in parsed:
        for j in i:
            try:
                links.append(j['links'][0]['href'])
            except:
                links.append('')
    #Dates
    for i in parsed:
        for j in i:
            try:
                dates.append(j['updated_parsed'])
            except:
                dates.append('')
    dates_days = []
    dates = [i[0:3] for i in dates]
    #Day of the month
    for i in dates:
        try:
            dates_days.append(i[2])
        except:
            dates_days.append('')
    #Make dataframe, limit to today
    today_central = int(timezone('US/Central').localize(datetime.now()).strftime('%d'))
    master = pd.DataFrame({'Links' : links, 'Dates' : dates_days})
    master = master.loc[master['Dates'] == today_central]
    master = master.reset_index(drop=True)
    return master

#Get text for articles
def get_text():
    master = get_urls()
    alco = AlchemyLanguageV1(api_key = os.environ.get('IBM_KEY'))
    all_articles = []
    j=1
    for i in master['Links']:
        try:
            text = alco.text(url=i)
            all_articles.append(text['text'])
            print "Done %s of %s" % (j, len(master['Links']))
            j+=1
            time.sleep(1)
        except:
            print ":("
            j+=1
            all_articles.append('')
    master['Text'] = all_articles
    return master

#Predict on model
def get_probabilities():
    master = get_text()
    model_trained = pickle.load(open('trained_model_new.p', 'rb'))
    predictions = model_trained.predict_proba(master['Text'])
    predict_list = [i[1][1] for i in enumerate(predictions)]
    master['Probabilities'] = predict_list
    return master

#Weight model
def define_weights():
    master = get_probabilities()
    weights_data = pd.DataFrame({'Frequency' : {'theawl.' : 246,
                                   'thehairpin.' : 36,
                                   'thebillfold.' : 12,
                                   'psmag.' : 1,
                                   'polygon.' : 86,
                                   'arstechnica.' : 30,
                                   'politico.' : 199,
                                   'fivethirtyeight.' : 31,
                                   'nytimes.' : 288,
                                   'thedailybeast' : 58,
                                   'citylab.' : 63,
                                   'newyorker.' : 166,
                                   'motherboard.' : 40,
                                   'atlasobscura.' : 42,
                                   'digiday.' : 18,
                                   'buzzfeed.' : 33,
                                   'longreads.' : 6,
                                   'newrepublic.' : 2,
                                    'theatlantic.': 126,
                                    'niemanlab.' : 48}})
    weights_data['Percent'] = weights_data['Frequency'] / sum(weights_data['Frequency'])
    weeks = [111, 111, 111, 109, 111, 109, 55, 104, 55, 108, 31, 55, 29, 55, 55, 55, 8, 55, 3, 2]
    weights_data['Probability'] = [master[master['Links'].str.contains(i)].mean().values[0] for i in feednames]
    weights_data['Weights'] = weights_data['Percent'] * weights_data['Probability']
    weights_data = weights_data.sort_values('Weights', ascending=False)
    weights_data['Weeks'] = weeks
    weights_data['Articles per Week'] = weights_data['Frequency'] / weights_data['Weeks']
    weights_data['Time Percent'] = weights_data['Articles per Week'] / sum(weights_data['Articles per Week'])
    weights_data['Time Weights'] = weights_data['Time Percent'] * weights_data['Probability'] * 2
    weights_data = weights_data.sort_values('Time Weights', ascending=False)
    return weights_data, master

def apply_weights():
    weights_data, master = define_weights()
    for i, row in master.iterrows():
        weighted_probability = 0
        for j in feednames:
            if j in row['Links']:
                weight = weights_data.loc[j]
                weighted_probability = row['Probabilities'] + row['Probabilities'] * weight['Time Weights']
        master.set_value(i, 'Weighted Probabilities', weighted_probability)
    master = master.sort_values('Weighted Probabilities', ascending=False)
    return master

def save_articles():
    master = apply_weights()
    savelinks = list(master['Links'].head(15))
    consumer_key = os.environ.get('POCKET_CONSUMER')
    access_token = os.environ.get('POCKET_ACCESS')
    pocket_instance = pocket.Pocket(consumer_key, access_token)
    #Save articles
    for i in savelinks:
        pocket_instance.add(i)
        print "article saved"

save_articles()
