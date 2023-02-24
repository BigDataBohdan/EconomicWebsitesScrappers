# Import necessary libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define the URL of the web page to scrape
url = "https://www.example.com"

# Send a GET request to the URL
response = requests.get(url)

# Get the content of the response
html_content = response.content

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")

# Find a single tag by tag name
single_tag = soup.find("tag_name")

# Find a single tag by tag name and attributes
single_tag_attr = soup.find("tag_name", attrs={"attr_name": "attr_value"})

# Find all tags by tag name
all_tags = soup.find_all("tag_name")

# Find all tags by tag name and attributes
all_tags_attr = soup.find_all("tag_name", attrs={"attr_name": "attr_value"})

# Find a tag by CSS class
tag_by_class = soup.find("tag_name", class_="class_name")

# Find all tags by CSS class
tags_by_class = soup.find_all("tag_name", class_="class_name")

# Find a tag by ID
tag_by_id = soup.find("tag_name", id="id_name")

# Find all tags by attribute
tags_by_attr = soup.find_all("tag_name", attrs={"attr_name": "attr_value"})

# Get the text of a tag
tag_text = tag.get_text()

# Get the value of an attribute of a tag
attr_value = tag["attr_name"]

# Get the value of a specific attribute using get method
attr_value_get = tag.get("attr_name")

# Get the parent tag of a tag
parent_tag = tag.parent

# Get the siblings of a tag
siblings = tag.find_next_siblings()

# Find the next sibling of a tag
next_sibling = tag.find_next_sibling()

# Find the previous sibling of a tag
prev_sibling = tag.find_previous_sibling()

# Find all tags within a specific HTML element
inner_tags = tag.find_all("tag_name")

# Find the next tag within a specific HTML element
next_tag = tag.find_next("tag_name")

# Find the previous tag within a specific HTML element
prev_tag = tag.find_previous("tag_name")

# Use regular expressions to find tags
import re

regex = re.compile("regex_pattern")
tags_by_regex = soup.find_all(regex)

# Save scraped data to a CSV file
data = {
    "column_name_1": [data_1],
    "column_name_2": [data_2],
    "column_name_3": [data_3]
}
df = pd.DataFrame(data)
df.to_csv("file_name.csv", index=False)

# Handle errors and exceptions
try:
    # Code that might raise an exception
except ExceptionType:
    # Code to handle the exception

# Handle HTTP errors
from requests.exceptions import HTTPError

try:
    response.raise_for_status()
except HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except Exception as err:
    print(f"Other error occurred: {err}")