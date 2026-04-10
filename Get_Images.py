import pandas as pd
import requests
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "GalaxyZoo1_DR_table2.csv"
IMAGE_DIR = "galaxy_images"
LIMIT = 2000  # How many images to download for training (Start small!)

# Create the folder if it doesn't exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# ==========================================
# LOAD DATA
# ==========================================
print(f"Loading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# ==========================================
# DOWNLOADER LOOP
# ==========================================
print(f"Starting download of {LIMIT} galaxy images from SDSS SkyServer...")

count = 0
for index, row in df.iterrows():
    if count >= LIMIT:
        break
    
    obj_id = str(row['OBJID'])
    ra = row['RA']
    dec = row['DEC']
    
    # File path: galaxy_images/123456.jpg
    file_path = os.path.join(IMAGE_DIR, f"{obj_id}.jpg")
    
    # Skip if we already have it
    if os.path.exists(file_path):
        count += 1
        continue
    
    # SDSS API URL (The "Telescope" Service)
    # We request a 64x64 pixel image at the galaxy's coordinates
    img_url = f"http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Chart.Image&ra={ra}&dec={dec}&scale=0.2&width=64&height=64"
    
    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"[{count+1}/{LIMIT}] Downloaded: {obj_id}")
            count += 1
        else:
            print(f"Failed to fetch {obj_id}: Status {response.status_code}")
            
    except Exception as e:
        print(f"Error downloading {obj_id}: {e}")
    
    # Be polite to the server (don't get banned!)
    time.sleep(0.5)

print("\n✅ Download Complete! Check the 'galaxy_images' folder.")