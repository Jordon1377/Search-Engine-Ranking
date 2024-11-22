class ScoringParams:
    def __init__(self, query_terms, bm25_score, document_metadata, link_data, weights=None):
        """
        Used for Combine Scores

        Parameters:
            query_terms: list
                List of query terms.
            bm25_score: float
                BM25 score for the document.
            document_metadata: dict
                Metadata for the document.
            link_data: dict
                Link analysis data for the document.
            weights: dict
                Tunable weights for scoring components and subcomponents.
        """
        self.query_terms = query_terms
        self.bm25_score = bm25_score
        self.document_metadata = document_metadata
        self.link_data = link_data
        self.weights = weights or {
            "bm25": 1.0,
            "metadata": {
                "freshness": 1.0,
                "docType": 1.0
            },
            "link_analysis": {
                "pageRank": 1.0,
                "inLinkCount": 0.5,
                "outLinkCount": -0.1
            },
        }
