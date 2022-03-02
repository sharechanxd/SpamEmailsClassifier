import re
import os
import random

filename = 'SMSSpamCollection.txt'
with open(filename, 'r',encoding = 'utf-8') as f:
    data = f.readlines()
f.close()

print(len(data))
test_index = random.sample(range(len(data)), 100)
test_data = [data[i] for i in test_index]

for i in sorted(test_index, reverse=True):
    del data[i]
print(len(test_data))
print(len(data))

tr = open('train.txt','w+',encoding = 'utf-8')
te = open('test.txt','w+',encoding = 'utf-8')
tr.writelines(data)
te.writelines(test_data)
tr.close()
te.close()


