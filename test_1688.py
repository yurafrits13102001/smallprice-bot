from curl_cffi import requests
from bs4 import BeautifulSoup

# Mobile version
url = "https://m.1688.com/offer/783370703296.html"
r = requests.get(url, impersonate="chrome120", timeout=15)
print("Mobile - Status:", r.status_code)
print("Mobile - Length:", len(r.text))
soup = BeautifulSoup(r.text, "html.parser")
print("Mobile - Title:", soup.title.string if soup.title else "NONE")
text = soup.get_text(separator="\n", strip=True)
lines = [l for l in text.split("\n") if len(l) > 10]
print("Mobile - Text lines:", len(lines))
for line in lines[:10]:
    print(" ", line[:150])

print("\n--- API TEST ---")
api_url = f"https://h5api.m.1688.com/h5/mtop.1688.trade.offer.model/1.0/?offerId=783370703296"
r2 = requests.get(api_url, impersonate="chrome120", timeout=15)
print("API - Status:", r2.status_code)
print("API - Length:", len(r2.text))
print("API - First 500:", r2.text[:500])
