from urllib.parse import quote  # For URL encoding
from utils import *
 
 
def get_url(species: str):
    species = remove_psithyrus(species)
    encoded = quote(species)
    return (
        f'https://library.big-bee.net/portal/collections/list.php?'
        f'hasimages=1&taxa={encoded}&usethes=1&taxontype=2'
        f'&association-type=none&comingFrom=newsearch&page=1'
    )
 
 
def main():
    species = get_species()
 
    urls = []
    for sp in species:
        urls.append(get_url(sp) + '\n')
 
    with open('urls.txt', 'w') as f:
        f.writelines(urls)
 
 
if __name__ == '__main__':
    main()