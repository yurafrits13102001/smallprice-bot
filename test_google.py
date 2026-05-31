import httpx
from bs4 import BeautifulSoup

url = "https://translate.google.com/translate?sl=zh-CN&tl=en&u=https://https://detail.1688.com/offer/783370703296.html?offerId=783370703296&spm=a260k.home2025.recommendpart.1"
r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, follow_redirects=True, timeout=15)
soup = BeautifulSoup(r.text, "html.parser")

# Show all text content
text = soup.get_text(separator="\n", strip=True)
lines = [l for l in text.split("\n") if len(l) > 10]
print("Non-empty lines:", len(lines))
for line in lines[:30]:
    print(line[:150])