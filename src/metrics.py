
def send_metrics(query, ranked_Document_Data):

    """
    Function for sending metrics over to evaluation

    Parameters:
        query: str
            The search query entered by the user.
        list of dict
            Ranked documents list with metadata and final scores.
        
    Returns:
        bool
            Did the data send over to evaluation sucessfully?
    """

    return True