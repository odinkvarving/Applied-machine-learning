import numpy as np

class Test():
    emojies = {
        'hat': '',
        'rat': 'U+1F400	',
        'cat': 'U+1F408	',
        'flat': 'U+1F3E2',
        'matt': 'U+1F9D1',
        'cap': 'U+1F9E2	',
        'son': 'U+1F466'
    }

    print("Hello world")

    emoji = np.eye(len(emojies))

    print(emoji)

    print("FFFFFFFFFFF")

    print(emoji[0])
