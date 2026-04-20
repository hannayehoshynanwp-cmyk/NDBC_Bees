# NBDC_Public
 
## Dependencies
To install required packages:
```
pip install -r requirements.txt
```

## Workflow
Everything is configured to work on Ecdysis, so it will have to be modified

`make_urls.py` should be run first
- This makes a URL for each species, writing them to `urls.txt`

`scrape.py` should be run next
- This retrieves URLs for every image found for each species, and puts them into the appropriate `<species>.txt` file within the `image_urls` folder

`download.py` should be run next
- This downloads every image using the URLs within `image_urls` and puts them into their respective species folders

## Other notes
The image URLs retrieved from scraping are the small thumbnails found in the table. As such they all have the suffix `_tn`. I assume this will be true for the new website as well. I assume the `_tn` means "tiny" because removing it leads to a much larger image, which we can actually use. `download.py` already removes the `_tn` portion.
- However, this results from many images being unretrievable: around 2000 of 11000 of the image URLs for Ecdysis led to 404 pages.
- As such, there may be a better way to retrieve the image URLs during scraping that leads to a higher image yield.