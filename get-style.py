import requests

url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl"
output_path = "stylegan2-ffhq-config-f.pkl"

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("✅ Download complete!")