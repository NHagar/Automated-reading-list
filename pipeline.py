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
    #Links and dates
    for i in parsed:
        for j in i:
            try:
                if j['updated_parsed'] == None:
                    pass
                else:
                    dates.append(j['updated_parsed'])
                    links.append(j['links'][0]['href'])
            except:
                pass

    dates_days = []
    dates = [i[0:3] for i in dates]
    #Day of the month
    for i in dates:
        dates_days.append(i[2])
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

def save_articles():
    master = get_probabilities().sample(n=15, weights=master['Probabilities'])
    savelinks = list(master['Links'])
    consumer_key = os.environ.get('POCKET_CONSUMER')
    access_token = os.environ.get('POCKET_ACCESS')
    pocket_instance = pocket.Pocket(consumer_key, access_token)
    #Save articles
    for i in savelinks:
        pocket_instance.add(i)
        print "article saved"

save_articles()
