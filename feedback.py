#Preserving article features and feedback
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_pickle('/users/nicholashagar/pocket/savedframe_2_15.p')
savelist = list(df['Links'].head(15))
preserve = pd.DataFrame(df['Text'].head(15))
choices = []
for i in savelist:
    print 'Did you enjoy %s' % i
    pref = raw_input('> ')
    if pref == 'y':
        print ':)'
        choices.append(1)
    elif pref == 'n':
        print ':('
        choices.append(0)
    else:
        print ':|'
        choices.append(0)
preserve['Liked'] = choices

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(preserve['Text'])
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
preserve['Features'] = X_train_tf
print preserve
