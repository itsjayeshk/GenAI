import tiktoken 


encoder = tiktoken.encoding_for_model('gpt-4o')

print("Vocab size", encoder.n_vocab)

text = "The cat is on the mat"
tokens = encoder.encode(text)
print("Tokens:", tokens)

my_tokens = [976, 9059, 382, 402, 290, 2450]
decoded = encoder.decode([976, 9059, 382, 402, 290, 2450])
print("Decoded:", decoded)