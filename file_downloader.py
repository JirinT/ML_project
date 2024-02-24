import requests
import os
import time
import zipfile

from tqdm import tqdm
from urllib.parse import urljoin


def read_list_from_file(file_path):
    # Initialize an empty list to store the strings
    string_list = []

    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Read each line from the file
        for line in file:
            # Remove leading and trailing whitespace characters (e.g., newline characters)
            line = line.strip()
            # Remove comma at the end of the line if present
            if line.endswith(','):
                line = line[:-1]
            # Append the line (string) to the string_list
            string_list.append(line)

    return string_list

def get_filename_from_url(url):
    # Send a HEAD request to the URL to retrieve headers only, but allow redirects
    response = requests.head(url, allow_redirects=True)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Check if the 'Content-Disposition' header exists
        if 'Content-Disposition' in response.headers:
            # Extract filename from the 'Content-Disposition' header
            content_disposition = response.headers['Content-Disposition']
            filename_start_index = content_disposition.find('filename=') + len('filename=')
            filename = content_disposition[filename_start_index:].strip('"')
            return filename
        else:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = "download_" + timestr
            return filename
    else:
        print(f"Failed to fetch headers for {url}. Status code: {response.status_code}")

    return None

def download_and_extract_file(url, download_dir):
    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Get the filename from the URL
    filename = get_filename_from_url(url)
    if filename is None:
        return

    if not filename.endswith('.zip'):
        filename += '.zip'

    # Send a GET request to the URL, but allow redirects
    response = requests.get(url, stream=True, allow_redirects=True)
    print(f"Download of {filename} started!")

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the total file size in bytes
        total_size = int(response.headers.get('content-length', 0))

        # Initialize tqdm with the total file size
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

        # Save the response content to a file
        with open(os.path.join(download_dir, filename), 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        progress_bar.close()
        print(f"\n{filename} downloaded successfully")

        # Extract the downloaded zip file
        with zipfile.ZipFile(os.path.join(download_dir, filename), 'r') as zip_ref:
            zip_ref.extractall(download_dir)

        # Remove the zip file
        os.remove(os.path.join(download_dir, filename))
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")


# Example usage:
download_dir = './caxton_dataset'  # Change this to the directory where you want to save the file

url_list = read_list_from_file("./hyperlinks.txt")

for url in url_list:
    download_and_extract_file(url, download_dir)
