

```python
from urllib2 import Request, urlopen, HTTPError
from urlparse import urlunparse, urlparse
import json 
import pandas as pd
import requests
import requests_cache
from bs4 import BeautifulSoup

import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
```

### NYT Analysis


```python
requests_cache.install_cache("project141")
```


```python
def geturl(timeperiod):
    """
    Get the article urls within a time period in NYT
    Input:a time period
    Output: a list of all the articles' urls within the period.
    """
    url="https://api.nytimes.com/svc/mostpopular/v2/mostviewed/all-sections/"+str(timeperiod)+"/?api-key=YourKey"
    response=requests.get(url)
    urllist=[response.json()[u'results'][i][u'url'] for i in range(len(response.json()[u'results']))]
    return urllist

N_urllist=geturl(7)
```


```python
def getcontent_NYT(url):
    """
    Extract text of an NYT article
    Input: url-> url for the article. 
    Output: a string of the text.      
    """
    artic= requests.get(url,"lxml")
    soup_arti=BeautifulSoup(artic.text)
    content=soup_arti.select("article[id='story'] > div[class='story-body-supplemental'] > div > p ")
    if content==[]:
        content=soup_arti.select("p[class='paragraph--story']")
    text=" ".join([i.text.strip() for i in content])
    return text
```


```python
a=' '.join([getcontent_NYT(link) for link in N_urllist])                #join all the text in a string
```


```python
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from scipy.misc import imread

t1_mask = imread("trumpb.jpg")
#wordcloud = WordCloud().generate(a)
wordcloud = WordCloud(max_words=200, mask=t1_mask, background_color='white').generate(a)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```


![png](output_6_0.png)


### the Guardian Analysis


```python
requests_cache.install_cache("project141G")
```


```python
def url_perpage(page,pagesize='50',fromdate='2017-02-22',todate='2017-02-28',form='json',order='oldest',key='YourKey'):
    """
    Get the urls for at most 50 articles from the Guardian in the last week of February
    Input: page number of the API content from which to extract the urls
    Output: a list of at most 50 urls.
    """
    url='https://content.guardianapis.com/search'
    response=requests.get(url,params={'page':page,'page-size':pagesize,'from-date':fromdate,'to-date':todate,'format':form,'order-by':order,'api-key':key})
    return [x['webUrl'] for x in response.json()['response']['results']]

urllist=[]
for page in range(1,37):
    urllist=urllist+url_perpage(page)
```


```python
def getcontent_Guardian(url):
    """
    Extract text of a the Guardian article
    Input: url-> url for the article. 
    Output: a string of the text.       
    """
    artic= requests.get(url)
    soup=BeautifulSoup(artic.text)
    try:
        text=" ".join([i.text.strip() for i in soup.find_all('div',itemprop='articleBody')[0].find_all('p')])
        text=text.translate({ 0x2018:0x27, 0x2019:0x27, 0x201C:0x22, 0x201D:0x22, 0x2026:0x20 })
    except (IndexError,TypeError):
        text=''
    return text
```


```python
a=' '.join([getcontent_Guardian(link) for link in urllist])                        #join all the text in a string
```


```python
stopwords=set(STOPWORDS)
stopwords.add("said")

t1_mask = imread("111.jpg")
wordcloud = WordCloud(max_words=100, mask=t1_mask,stopwords=stopwords,background_color='white').generate(a)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```


![png](output_12_0.png)


### Similar articles


```python
stemmer = PorterStemmer().stem
tokenize = nltk.word_tokenize
def stem(tokens,stemmer = PorterStemmer().stem):
    """
    stemmatize words
    Input: words
    Output: stemmer
    """
    return [stemmer(w.lower()) for w in tokens] 

def lemmatize(text):
    """
    Extract simple lemmas based on tokenization and stemming
    Input: string
    Output: list of strings (lemmata)
    """
    return stem(tokenize(text))
```


```python
NYT=pd.DataFrame()                                                   #convert all the NYT article texts into a dataframe
for link in N_urllist:
    NYT=NYT.append(pd.DataFrame({'text':[getcontent_NYT(link)]}),ignore_index=True)
```


```python
Guardian=pd.DataFrame()                                              #convert all the Guardian article texts into a dataframe
for link in urllist:
    Guardian=Guardian.append(pd.DataFrame({'text':[getcontent_Guardian(link)]}),ignore_index=True)
```


```python
total=Guardian.append(NYT,ignore_index=True)                         #merge the two dataframes together
```

__Then we get the most similar articles by similarity matrix__


```python
vectorizer = TfidfVectorizer(tokenizer=lemmatize,stop_words="english",smooth_idf=True,norm=None)
A = vectorizer.fit_transform(total['text'])
```


```python
P=A.toarray()
K=P.dot(P.T)
U=np.triu(K,k=1)
```


```python
V=U[0:1783,1783:]
```


```python
sort=np.sort(V,axis=None)
print np.where(V==sort[-1]), np.where(V==sort[-2]), np.where(V==sort[-3])
```

    (array([30], dtype=int64), array([4], dtype=int64)) (array([30], dtype=int64), array([9], dtype=int64)) (array([1698], dtype=int64), array([4], dtype=int64))
    


```python
 getcontent_NYT(N_urllist[4])
```




    u'I spent last week ignoring President Trump. Although I am ordinarily a politics junkie, I didn\u2019t read, watch or listen to a single story about anything having to do with our 45th president. What I missed, by many accounts, was one of the strangest and most unpredictable weeks of news in modern political history. Among other things, there was the resignation of the national security adviser, Michael T. Flynn, and an \u201cOprah Winfrey Show\u201d tape that led to the downfall of the nominee for secretary of labor, Andrew F. Puzder. It wasn\u2019t my aim to stick my head in the sand. I did not quit the news. Instead, I spent as much time as I normally do online (all my waking hours), but shifted most of my energy to looking for Trump-free zones. My point: I wanted to see what I could learn about the modern news media by looking at how thoroughly Mr. Trump had subsumed it. In one way, my experiment failed: I could find almost no Trump-free part of the press. But as the week wore on, I discovered several truths about our digital media ecosystem. Coverage of Mr. Trump may eclipse that of any single human being ever. The reasons have as much to do with him as the way social media amplifies every big story until it swallows the world. And as important as covering the president may be, I began to wonder if we were overdosing on Trump news, to the exclusion of everything else. The new president doesn\u2019t simply dominate national and political news. During my week of attempted Trump abstinence, I noticed something deeper: He has taken up semipermanent residence on every outlet of any kind, political or not. He is no longer just the message. In many cases, he has become the medium, the ether through which all other stories flow. Obviously, just about every corner of the news was a minefield, but it was my intention to keep informed while avoiding Mr. Trump. I still consulted major news sites, but avoided sections that tend to be Trump-soaked, and averted my eyes as I scrolled for non-Trump news. I spent more time on international news sites like the BBC, and searched for subject-specific sites covering topics like science and finance. I consulted social news sites like Digg and Reddit, and occasionally checked Twitter and Facebook, but I often had to furiously scroll past all of the Trump posts. (Some news was unavoidable; when Mr. Flynn resigned, a journalist friend texted me about it.) Even when I found non-Trump news, though, much of it was interleaved with Trump news, so the overall effect was something like trying to bite into a fruit-and-nut cake without getting any fruit or nuts. It wasn\u2019t just news. Mr. Trump\u2019s presence looms over much more. There he is off in the wings of \u201cThe Bachelor\u201d and even \u201cThe Big Bang Theory,\u201d whose creator, Chuck Lorre, has taken to inserting anti-Trump messages in the closing credits. Want to watch an awards show? Say the Grammys or the Golden Globes? Trump Trump Trump. How about sports? Yeah, no. The president\u2019s policies are an animating force in the N.B.A. He was the subtext of the Super Bowl: both the game and the commercials, and maybe even the halftime show. Where else could I go? Snapchat and Instagram were relatively safe, but the president still popped up. Even Amazon.com suggested I consider Trump toilet paper for my wife\u2019s Valentine\u2019s Day present. (I bought her jewelry.) All presidents are omnipresent. But it is likely that no living person in history has ever been as famous as Mr. Trump is right now. It\u2019s possible that not even the most famous or infamous people of the recent or distant past \u2014 say, Barack Obama, Osama bin Laden, Bill Clinton, Richard Nixon, Michael Jackson, Muhammad Ali or Adolf Hitler \u2014 dominated media as thoroughly at their peak as Mr. Trump does now. I\u2019m hedging because there isn\u2019t data to directly verify this declaration. (Of course, there are no media analytics to measure how many outlets were covering Hitler the day he invaded Poland.) But there is some pretty good circumstantial evidence. Consider data from mediaQuant, a firm that measures \u201cearned media,\u201d which is all coverage that isn\u2019t paid advertising. To calculate a dollar value of earned media, it first counts every mention of a particular brand or personality in just about any outlet, from blogs to Twitter to the evening news to The New York Times. Then it estimates how much the mentions would cost if someone were to pay for them as advertising. In January, Mr. Trump broke mediaQuant\u2019s records. In a single month, he received $817 million in coverage, higher than any single person has ever received in the four years that mediaQuant has been analyzing the media, according to Paul Senatori, the company\u2019s chief analytics officer. For much of the past four years, Mr. Obama\u2019s monthly earned media value hovered around $200 million to $500 million. The highest that Hillary Clinton got during the presidential campaign was $430 million, in July. It\u2019s not just that Mr. Trump\u2019s coverage beats anyone else\u2019s. He is now beating pretty much everyone else put together. Mr. Senatori recently added up the coverage value of 1,000 of the world\u2019s best known figures, excluding Mr. Obama and Mr. Trump. The list includes Mrs. Clinton, who in January got $200 million in coverage, Tom Brady ($38 million), Kim Kardashian ($36 million), and Vladimir V. Putin ($30 million), all the way down to the 1,000th most-mentioned celebrity in mediaQuant\u2019s database, the actress Madeleine Stowe ($1,001). The coverage those 1,000 people garnered last month totaled $721 million. In other words, Mr. Trump gets about $100 million more in coverage than the next 1,000 famous people put together. And he is on track to match or beat his January record in February, according to Mr. Senatori\u2019s preliminary figures. How do we know Mr. Trump is more talked about than anyone else in the past? There are now more people on the planet who are more connected than ever before. Facebook estimates that about 3.2 billion people have internet connections. On average, the people of Earth spend about eight hours a day consuming media, according to the marketing research firm Zenith. So almost by definition, anyone who dominates today\u2019s media is going to be read about, talked about and watched by more people than ever before. \u201cFrom a media perspective, it\u2019s pretty clear,\u201d Mr. Senatori said. \u201cThe sheer volume, and the sheer amount of consumption, and all the new channels that are available today show that, yeah, he\u2019s off the charts.\u201d Mr. Trump is a historically unusual president, and thus deserves plenty of coverage. Yet there\u2019s an argument that our tech-fueled modern media ecosystem is amplifying his presence even beyond what\u2019s called for. On most days, Mr. Trump is 90 percent of the news on my Twitter and Facebook feeds, and probably yours, too. But he\u2019s not 90 percent of what\u2019s important in the world. During my break from Trump news, I found rich coverage veins that aren\u2019t getting social play. ISIS is retreating across Iraq and Syria. Brazil seems on the verge of chaos. A large ice shelf in Antarctica is close to full break. Scientists may have discovered a new continent submerged under the ocean near Australia. There\u2019s a reason you aren\u2019t seeing these stories splashed across the news. Unlike old-school media, today\u2019s media works according to social feedback loops. Every story that shows any signs of life on Facebook or Twitter is copied endlessly by every outlet, becoming unavoidable. Scholars have long predicted that social media might alter how we choose cultural products. In 2006, Duncan Watts, a researcher at Microsoft who studies social networks, and two colleagues published a study arguing that social signals create a kind of \u201cinequality\u201d in how we choose media. The researchers demonstrated this with an online market for music downloads. Half of the people who arrived at Mr. Watts\u2019s music-downloading site were shown just the titles and band name of each song. The other half were also shown a social signal \u2014 how many times each song had been downloaded by other users. Mr. Watts and his colleagues found that adding social signals changed the music people were interested in. Inequality went up: When people could see what others were downloading, popular songs became far more popular, and unpopular songs far less popular. Social signals also created a greater unpredictability of outcomes; when people could see how others had picked songs, the collective ratings of each song were less likely to predict success, and bad songs were more likely to become popular. I suspect we are seeing something like this effect playing out with Trump news. It\u2019s not that coverage of the new administration is unimportant. It clearly is. But social signals \u2014 likes, retweets and more \u2014 are amplifying it. Every new story prompts outrage, which puts the stories higher in your feed, which prompts more coverage, which encourages more talk, and on and on. We saw this effect before Mr. Trump came on the scene \u2014 it\u2019s why you know about Cecil the lion and Harambe the gorilla \u2014 but he has accelerated the trend. He is the Harambe of politics, the undisputed king of all media. It\u2019s only been a month since Mr. Trump took office, and already the deluge of news has been overwhelming. Everyone \u2014 reporters, producers, anchors, protesters, people in the administration and consumers of news \u2014 has been amped up to 11. For now, this might be all right. It\u2019s important to pay attention to the federal government when big things are happening. But Mr. Trump is likely to be president for at least the next four years. And it\u2019s probably not a good idea for just about all of our news to be focused on a single subject for that long. In previous media eras, the news was able to find a sensible balance even when huge events were preoccupying the world. Newspapers from World War I and II were filled with stories far afield from the war. Today\u2019s newspapers are also full of non-Trump articles, but many of us aren\u2019t reading newspapers anymore. We\u2019re reading Facebook and watching cable, and there, Mr. Trump is all anyone talks about, to the exclusion of almost all else. There\u2019s no easy way out of this fix. But as big as Mr. Trump is, he\u2019s not everything \u2014 and it\u2019d be nice to find a way for the media ecosystem to recognize that.'




```python
getcontent_Guardian(urllist[30])
```




    u'Now that political satire is dead, we must make some decisions about what will occupy the space in our lives it used to fill. You might just shrug and say "events". And while it\'s true that most mornings the radio news bulletin serves up as scabrous an indictment of politics as anything a writer could invent, it\'s not funny like Veep. Some weird synthesis of the real and the made-up \u2013 you can use the term "evention" if you promise not to credit me \u2013 may be the answer. But while you\'re waiting for it to come along, have a look the Twitter account @RoguePOTUSStaff. It purports to be leaking information from inside the White House, mostly more in sorrow than in anger: "POTUS was dismissive of PM May. \'I hope she\'ll like us, but she doesn\'t have much to offer\'." After less than a week, the account has approaching half a million followers. According to Twitter, this includes 57 people I follow \u2013 a thin but fairly representative cross section of the sneering liberal media elite. Taken together, the account\'s 200-odd tweets paint a picture of a scheming and incompetent inner circle desperately trying to rein in a president who has little understanding of his role and few interests beyond gratifying his own vanity. And that rings true, because it\'s exactly what dismayed liberals like me want to hear. The account is one of dozens of "unofficial" \u2013 and in some cases fake \u2013 Twitter accounts set up after Trump cracked down on tweeting across government agencies. But it is crafted with some care, and so far has avoided any obvious pitfalls of fakery. Enough @LouiseMensch . Why don\'t you tell the truth, that you\'re angry at us for not divulging our IDs for your website. THAT is Bannon-esk. pic.twitter.com/XhziE2inRf Of course part of its appeal is that it might yet be real. It tracks real-time events so closely that it seems to have the knack of prescience. Theories about its inauthenticity (including claims it is a distraction created by White House adviser Steve Bannon) are, if anything, more preposterous. One debunker alleged it was Russian propaganda because one tweet used the word "Bannon-esk" instead of "Bannonesque". I\'m no linguist, but I\'d guess the odd spelling was deployed to save space in a 140-character tweet. Most of all @RoguePOTUSStaff offers hope in the guise of despair \u2013 the promise that an administrative setup this stupid cannot endure. There\'s one thing it certainly isn\'t, and that\'s satire. There are no jokes, and nothing like exaggeration. If isn\'t real, it might as well be. A new study in the journal Nature Neuroscience attests to the efficacy of "overlearning" \u2013 the strategy of continuing to practise something one has already mastered. Even if you don\'t appear to be getting better at something, you are still helping to cement the skill in your brain, so that it\'s harder for new learning to come along and dislodge it. Such "hyperstabilisation" is associated with an abrupt shift from glutamate-dominant excitatory to GABA-dominant inhibitory processing, but I don\'t have to tell you that. I\'m a latecomer to the idea. Once I\'ve mastered a new skill I lose all interest in it, which is why one of the doors of our hall cupboard opens smoothly, and the other one still comes off in your hand. But I am familiar with new stuff threatening to push old stuff out of my brain. This gets worse with age, so that I now have to overlearn a postcode \u2013 by chanting it to myself \u2013 in order to hang on to it long enough to type it into my phone.  Given the chaos, incompetence and infighting that may or may not characterise the present White House administration, what are the odds that Donald Trump will not last a full four years as president? My father-in-law took his wishful thinking down to Ladbrokes, and put some cash down on that question. He seemed a bit disappointed by the answer. "They only gave me even money," he said.'


