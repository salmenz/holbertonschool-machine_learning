#!/usr/bin/env python3
"""find a snippet of text within a reference document to answer a q"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    q = input('Q: ')
    while(q.lower() not in ['exit', 'quit', 'goodbye', 'bye']):
        answer = question_answer(q, reference)
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A:", answer)
        q = input('Q: ')
    print('A: Goodbye')
