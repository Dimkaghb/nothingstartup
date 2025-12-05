import requests
from bs4 import BeautifulSoup

def scrape_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator="\n", strip=True)

default_url = "https://en.wikipedia.org/wiki/Cars_(film)"
user_input = input(f"Enter a URL (press Enter for default: {default_url}): ").strip()

url = user_input if user_input else default_url

content = scrape_page(url)
print(content)
