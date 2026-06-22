import os
import urllib.request

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())
        print("Successfully downloaded!")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    os.makedirs("fonts", exist_ok=True)
    download_file("https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf", "fonts/Inter-Regular.ttf")
    download_file("https://raw.githubusercontent.com/google/fonts/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf", "fonts/Inter-Bold.ttf")


