"""This file is used for running scripts
"""
import sys
import asyncio

from ucla_topic_analysis.data.coroutines.lda import LdaPipeline

def get_number_of_topics():
    """Gets the number of topics via user input

    Returns:
        int: The number of topics to use in the LDA model
    """
    num_topics = None

    # Try to set num_topics from command line arguments
    try:
        num_topics = int(sys.argv[1])
        if num_topics <= 0:
            raise ValueError()
    except (IndexError, ValueError) as error:
        if isinstance(error, ValueError):
            print("Invalid command line argument for number of topics.", end=" ")
            print("Value must be an integer greater than 0,", end=" ")
            print("got '{0}' instead".format(sys.argv[1]))
        num_topics = None

    # Get num_topics from the user
    while not num_topics:
        try:
            user_input = input("Enter number of topics: ")
            num_topics = int(user_input)
            if num_topics <= 0:
                raise ValueError()
        except ValueError:
            print("Invalid input. Input must be an integer greater than 0", end=" ")
            print("got '{0}' instead".format(user_input))
            num_topics = None
    return num_topics

def main():
    """Runs the application
    """
    try:
        num_topics = int(sys.argv[1])
        if num_topics <= 0:
            raise ValueError()
    except (IndexError, ValueError):
        num_topics = get_number_of_topics()
    print("Training LDA model with {0} topics".format(num_topics))
    pipeline = LdaPipeline(num_topics)
    asyncio.run(pipeline.train())

if __name__ == "__main__":
    main()
