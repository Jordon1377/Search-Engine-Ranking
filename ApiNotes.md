These are the APIs factored in to only ranking, there are others for other operations

## Indexing
```py
getTotalDocStatistics(): Returns statistics of all documents in the database
{
      "avgDocLength": 798.8730,
      "docCount": 4567876
}
```
```py
getDocsFromIndex(term)
Returns a JSON containing the invertible index on all docs for the term
{
	"term": "data",
	"index": [
    	{
        	"docID": "12345",
        	"frequency": 5,
        	"positions": [4, 15, 28, 102, 204]
    	},
    	// More documents
	]
}
```
```py
getDocumentMetadata (docID)
Returns JSON document metadata
{
	"docID": "12345",
	"metadata": {
		"docLength": 2450,
        "timeLastUpdated": "2024-11-09T15:30:00Z",
        "docType": "PDF",
        "docTitle": "Introduction to Data Science",
        "URL": "https://example.com/documents/12345"
        // More metadata as generated...
	}
}
```

## Link Analysis
```py
getPageRank(URL)
Given a document ID or URL returns the Page Rank score for that specific link.
{
	"pageRank": 63.379
	"inLinkCount": 0
	"outLinkCount": 999999
}
```

