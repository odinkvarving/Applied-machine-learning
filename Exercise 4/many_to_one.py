import torch
import torch.nn as nn
import numpy as np


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, char_encodings_size, emoji_encodings_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(char_encodings_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, emoji_encodings_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a'
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't'
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r'
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'c'
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'f'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'm'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'p'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 's'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'n'
]

char_encodings_size = len(char_encodings)

# indexes        '0'  '1'  '2'  '3'  '4'  '5'  '6'  '7'  '8'  '9'  '10' '11' '12'
index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

x_train = torch.tensor([
    [[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],     # Matrix for 'hat '
    [[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],     # Matrix for 'rat '
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],     # Matrix for 'cat '
    [[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]],     # Matrix for 'flat'
    [[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]],     # Matrix for 'matt'
    [[char_encodings[5]], [char_encodings[2]], [char_encodings[9]], [char_encodings[0]]],     # Matrix for 'cap '
    [[char_encodings[10]], [char_encodings[11]], [char_encodings[12]], [char_encodings[0]]],  # Matrix for 'son '
], dtype=torch.float)

batches = 7
x_train_batches = torch.split(x_train, batches)

emoji_encodings = [
    [1., 0., 0., 0., 0., 0., 0.],  # '🎩'
    [0., 1., 0., 0., 0., 0., 0.],  # '🐁'
    [0., 0., 1., 0., 0., 0., 0.],  # '🐈️'
    [0., 0., 0., 1., 0., 0., 0.],  # '🏢'
    [0., 0., 0., 0., 1., 0., 0.],  # '👨'
    [0., 0., 0., 0., 0., 1., 0.],  # '🧢'
    [0., 0., 0., 0., 0., 0., 1.],  # '👶'
    ]

emoji_encodings_size = len(emoji_encodings)

index_to_emoji = ['🎩', '🐁', '🐈️', '🏢', '👨', '🧢', '👶']

y_train = torch.tensor([
    [emoji_encodings[0], emoji_encodings[0], emoji_encodings[0], emoji_encodings[0]],  # Matrix for '🎩'
    [emoji_encodings[1], emoji_encodings[1], emoji_encodings[1], emoji_encodings[1]],  # Matrix for '🐁'
    [emoji_encodings[2], emoji_encodings[2], emoji_encodings[2], emoji_encodings[2]],  # Matrix for '🐈'
    [emoji_encodings[3], emoji_encodings[3], emoji_encodings[3], emoji_encodings[3]],  # Matrix for '🏢'
    [emoji_encodings[4], emoji_encodings[4], emoji_encodings[4], emoji_encodings[4]],  # Matrix for '👨'
    [emoji_encodings[5], emoji_encodings[5], emoji_encodings[5], emoji_encodings[5]],  # Matrix for '🧢'
    [emoji_encodings[6], emoji_encodings[6], emoji_encodings[6], emoji_encodings[6]],  # Matrix for '👶'
], dtype=torch.float)

y_train_batches = torch.split(y_train, batches)

model = LongShortTermMemoryModel(char_encodings_size, emoji_encodings_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.0005)
for epoch in range(1000):
    for batch in range(7):  #Using number of words as range (7), so the model will be trained in batches
        model.reset()
        model.loss(x_train[batch], y_train[batch]).backward()
        optimizer.step()
        optimizer.zero_grad()



