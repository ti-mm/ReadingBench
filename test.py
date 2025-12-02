import urllib, urllib.request
url = 'http://export.arxiv.org/api/query?search_query=ti:attention_is_all_you_need&start=0&max_results=5'
data = urllib.request.urlopen(url)
print(data.read().decode('utf-8'))