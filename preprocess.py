import json
from time import time
from wiki_main_content_scraper import scrape_wiki
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
from nltk import word_tokenize, wordpunct_tokenize
from textblob import TextBlob
from autocorrect import Speller
from nltk.stem import SnowballStemmer
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

url = 'https://en.wikipedia.org/wiki/The_Mystery_of_the_Yeti,_Part_2'
# corpus = scrape_wiki(url)
# with open('korpus.txt', 'w', encoding='utf8') as f:
#     f.write(corpus)
with open('korpus.txt', 'r', encoding='utf8') as f:
    corpus = f.read()

# corpuss = corpus

start = time() # string: replace, lower, word_tokenize, remove stopwords
for c in punctuation + '\n':
    corpus = corpus.replace(c, '')

corpus = corpus.lower()
# speller = Speller() # za str
# corpus = speller(corpus) # masu sporo, mislin 30ak sek za ovu sicu od teksta; ispravilo smal i varius

stopwordz = stopwords.words('english') # 179
wordz = word_tokenize(corpus)
for word in wordz:
    if word in stopwordz:
        wordz.remove(word) # ako nije maklo a,the dodat

end = time()
print("String ops time: ", end - start)
####################################################################################################
# start = time() # df: replace, lower, word_tokenize, remove stopwords
# df = pd.DataFrame({'text':[corpuss]})
# for c in punctuation + '\n':
#     df['text'] = df['text'].str.replace(c, '', regex=False) # ovde =rez za TiF
#     # df = df.replace(c,'', regex=True)

# stopwordz = stopwords.words('english') # 179
#                                       #ˇovde iza lambda definiras varijablu word
# df['text'] = df['text'].apply(lambda word: " ".join(word.lower() for word in word.split() if word not in stopwordz))

# end = time()
# print("Dataframe ops time: ", end - start) # za ovako napisano i ovaj mini dataset string cca.50% brzi od pd
#############################################################################################################

df = pd.DataFrame({'text':wordz}) #489 rici
# df['text'] = df['text'].apply(lambda words: str(TextBlob(words).correct())) # attempt.;str da nebude (o,v,a,k,o)
                                                #==sporo ko blato(Speller), ==popravilo
# kratice-lookup_dict nista
# stemming                                  
sbs = SnowballStemmer('english')
df['text'] = [sbs.stem(wrd) for wrd in df['text']]
# print(df['text'])
# EDA
# frequency_distribution = FreqDist(df['text'])
# print(frequency_distribution.elements)
# frequency_distribution.plot()

################################################ test ####################################
# def is_engineer(education_field):
#     edu_field_cat = {"Technical degree" : "Engineer"}
#     return edu_field_cat.get(education_field, "Not an engineer") # get() dict metoda, vrati v za k ako ima inace default

# df["EducationFieldCategory"] = [is_engineer(x) for x in df['text']] # radi
################################################ endof test ################################
def is_long(word):
    return True if len(word) > 3 else False

# EDA, freq_dist, wcloud
dfl = pd.DataFrame()
dfl['text'] = [x for x in df['text'] if is_long(x)]
frequency_distribution = FreqDist(dfl['text'])
print(frequency_distribution.elements) # pd.series u sebi ima np.array 
# frequency_distribution.plot()
# json.dump(frequency_distribution,open(f'jaysson.json', 'w', encoding='utf8'))

wcloud = WordCloud().generate_from_frequencies(frequency_distribution) #ima stopwords u sebi
# plt.imshow(wcloud)
# plt.axis('off')
# plt.show()

################### one hot dummy ######################################
dummie = pd.get_dummies(dfl['text']) # one hot djir; vraca DataFrame
# print(dummie) # 392r, 251c; znaci dosta duplih - 251/392 unique
print(sum(dummie['yeti'])) # mos sumirat stupce za br ponavljanja rici
############################### Count Vect ################################################
vectorizer = CountVectorizer() # kreiranje transformacije
vectorizer.fit(dfl['text']) # tokenizacija;kreiramo/ucimo rjecnik svih tokena; izbaci interpunk. i kratke rici etc.
vector = vectorizer.transform(dfl['text']) # transformacija; DocumentTermMatrix
# print(vectorizer.vocabulary_)
# print(vector.toarray())
# print(vector) # ([0-br.rici], indexurijecniku(<, neke se ponavljaju))=br.pojavljivanja; vector[391,251]=0


################################## TF-IDF ###############################################
# TermFrequency u stringu, InverseDocumentFreq u cilon korpusu
# TF-omjer pojavljivanja rici i ukupno rici u JEDNOJ recenici/stringu/reviewuiguess
# txtstr = ["This movie is not scary and is slow", "This movie is very scary and long",
#         "This movie is spooky and good"]
#IDF ponistava beskorisne veznike itd. koji se pojavljuju svugdi
# IDF('this) = log(br.rec / br.rec di se 'this' pojavljuje) = log(3/3) = 0-bezvrijedna
# IDF('not') = log(3/1) = 0.48
# IDF('scary') = log(3/2) 0.18  # mal raspon [0,x]
# 3/3 1, 2/3 0.67, 1/3 0.33 bi bilo bez log obratno; za neka 2pr. 166x raz., a s log. samo 8x

#TF-IDF = TF * IDF # taj rez. ide u mrezu
#('this') = 1/8 * 0 = 0
#('not') = 1/8 * 0.48 = 0.06

# txtstr = ["This is some words that repeat, some words.", "Some other words and some the same",
#             "Even more words"]
##vectorizer = TfidfVectorizer()
# vectorizer.fit(dfl['text']) # tokenizacija i izgradnja vokabulara; opt. obrade, pretvori moj row-word-string u ['word']
##test = ["Three", "two", "one"]
##vectorizer.fit(test)
# print(vectorizer.vocabulary_)
##print(vectorizer.idf_)

# idf = ln[(1 + n)/(1 + df)]; tf = br.x u danom dokumentu
#### tfidf bi tribalo racunat na vise clanaka, ili bar paragrafa, ovako beskorisno
pass