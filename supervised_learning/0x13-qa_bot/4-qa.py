#!/usr/bin/env python3
"""answers questions from multiple reference texts"""
qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(coprus_path):
    """answers questions from multiple reference texts"""
    q = input('Q: ')
    while(q.lower() not in ['exit', 'quit', 'goodbye', 'bye']):
        reference = semantic_search(coprus_path, q)
        answer = qa(q, reference)
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A:", answer)
        q = input('Q: ')
    print('A: Goodbye')
