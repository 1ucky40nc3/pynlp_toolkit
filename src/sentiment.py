"""This module holds methods to do sentiment-analysis on a given text

Examples:
    import sentiment

    sentiment.analysis("I am very optimistic.")
    >> ...

    sentiment.analysis("This could have gone better.")
    >> ...
"""


from transformers import pipeline


def analysis(text):
    """Analize a given text.

    A given text will be analyzed via:
        - transfomers sentiment-analysis pipeline
        - method 2
        ...

    Args:
        text (str): The text that is to be analized.

    Returns:
        results (dict): A dictionary serving as dynamic 
                        wrapper for analysis results.
                        
                        The output could be simplified as:

                        {
                            method1: results1,
                            method2: results2,
                            ...
                        }
    """
    results = {}

    classifier = pipeline('sentiment-analysis')
    results["sentiment-analysis"] = classifier(text)

    return results
    