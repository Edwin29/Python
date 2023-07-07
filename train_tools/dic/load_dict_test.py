import pickle

f = open("chatbot_dict1.bin", "rb")
word_index = pickle.load(f)
f.close()

print(word_index['OOV'])
print(word_index['오늘'])
print(word_index['주문'])
print(word_index['탕수육'])
print(word_index['1시'])
