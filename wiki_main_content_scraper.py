import re
import requests
from bs4 import BeautifulSoup as bs

def scrape_wiki(url):
    response = requests.get(url)
    print(f"\nMain content of page: {response.url}\n")
    soup = bs(response.content, features="html.parser")

    main_content = soup.find('div', attrs={'class':'mw-parser-output'})
    paragraph = main_content.find_next('p') # opt. dodat <li>ste
    main_paragraphs = []
    while paragraph != None:
        text = re.sub(r'\[[^]]*\]', '', paragraph.text) # maknit [1] reference
        main_paragraphs.append(text)

        paragraph = paragraph.find_next('p')

    corpus = ' '.join(main_paragraphs)
    return corpus

if __name__ == '__main__':
    url = "https://en.wikipedia.org/wiki/Special:Random"
    paragraphs = scrape_wiki(url)

    for p in paragraphs:
        print(p)