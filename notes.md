Example URL for Apis mellifera

https://ecdysis.org/collections/list.php?db=133,151,82,87,2,1,3,172,125,178,159,102,169,158,104,55,155,111,78,76,191,15,89,27,37,24,120,105,132,149,41,84,54,81,63,177,17,180,170,198,129,122,118,138,157,53,34,90,67,66,80,202,79,160,69,36,62,42,146,101,126,96,207,114,115,95,119,7,109,56,93,203,131,205,128,106,148,92,83,123,140,52,73,136,74,38,166,199,100,107,161,135,163,143,91,10,110,124,182,94,98,164,211,112,171,150,19,139,153,29,32,121,88,86,8,201,71,60,156,51,13,108,97,30,33,61,152,145,85,26,4,142,116,154,204,147,65,196,45,39,16,130,75,9,20,21,18,167,175,195,127,197,113,194&hasimages=1&taxa=Apis%20mellifera&usethes=1&taxontype=2&association-type=none&comingFrom=newsearch&page=1

Only need to change `taxa` and `page`

# Page Structure
```
<table id="omlisttable">
  ...
    <td>
      <div>
        ...
          <img src="..." alt="Image Associated With the Occurrence">  // Consistent alt text
      </div>

      <div>
        ...
          <i> Species name</i>  // Always a leading blank
      </div>

      ...
```

Every img src ends with `_tn.jpg`, with the tn denoting tiny I believe. Removing the _tn leads to a large image


Searches seem to return only species with exactly matching names


Last page or empty queries will contain this element
`<h3>Your query did not return any results. Please modify your query parameters</h3>`


`https://ecdysis.org/images/image-icon.svg` is a placeholder img and can be discarded/not downloaded