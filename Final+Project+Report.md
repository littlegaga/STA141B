
<h1> Mr./ Trump/ Said <h1>
<p> The analysis of trending words in the last week of Feburary<p>
<p> Yuren Zhang, Zhihan Zhu <p>

<p>     Trending words are good indicators of the current status. Higher word frequencies are positively correlated to more coverage and exposure. In other words, more people are talking or searching about this term. To find out what are the trending words in the last week of Feburary, my partner and I looked at two mainstream media: The New York Times (NYT) in the U.S. and the Guardians in the U.K. We found that the trending words from the two media varies significantly.<p>

<h3> Part 1. The analysis of NYT most viewed articles</h3>

<p> NYT provides its API for public use. Thus, we utilize this advantage and acquired the links for most viewed articles in all sections. We choose not to download all articles within the month because of the limitation of non-commercial license as well as poor efficiency. Then, those links are used to access individual articles. The content of individual articles are acquired by web-scrapping. Text are stripped and analyzed after removing stopwords. The result is presented in Word-Cloud form. <p>

<img src="image.png">

<p> The results showed that in the last week of Feburary, Mr. Trump, the President clearly took the majority attention from NYT reporters and audiences. In addition, his opinions and statements are wildly reported as the word said appears a lot. <p>

<p> The U.S. is clearly focusing on Mr. Trump. However, across the Atlantic, people do not necessarily care about the same.<p>

<h3> Part 2. The analysis of the Guardian</h3>

The Guardian doesn't have a API method of most viewed articles that is similar to NYT, so we choose to obtained all the articles of the last week of Februrary. With similar skills of web-scrapping, text analysis, we get the following Word-Cloud:

<img width=400 src="download.png"><br>

Although we still can see Trump on that picture, it is not the most popular thing that British people care. From those words such as "year", "new" and "time", we can have a guess that the Guardian focuses on the future, the plan of the country since February was still the beginning of 2017.

<h3>Similar Articles</h3>

<p> Since the difference is obvious, we would also like to know are there any similar articles between the two media in two different countries. <p>

The two most similar articles from these two medias are __“I Ignored Trump News for a week. Here’s what I Learned”__ from NYT and __“I Ignored Trump News for a week. Here’s what I Learned”__ from the Guardian.
The first one talks about the phenomenon that Trump is everywhere on social medias, which exactly reflects the Word-Cloud.
The second one talks about an twitter account that purports to leak information of the president. 
So we can see that the intersection between the trends in this week was also President Trump.

<h3>For the next step</h3>

Because of poor computer performance, we campare the most popular content from NYT with all the articles from the Guardian, which might be inappropriate. For the same reason, we only investigae on one week.
Therefore, for the next step we want to comparing the contents from the two medias of a whole month (with almost 10,000 articles each). 
This could 
<ol>
<li>increasing accuracy.</li>
<li>potentially implementing an analysis of word frequency change against time.</li>
<ol>


```python

```
