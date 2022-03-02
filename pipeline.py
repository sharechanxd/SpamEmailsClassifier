from naiveBayes import *
from sklearn.metrics import confusion_matrix,classification_report

train_vp = VocProc('data/train.txt')
train_sms_words,train_labels = train_vp.data_loader()
train_voc = train_vp.creatVocVector()
vocabulary_list = train_vp.creatVocList()

model = naiveBayes(train_voc,train_labels,vocabulary_list)
model.fit()


test_vp = VocProc('data/test.txt')
test_sms_words,test_labels = test_vp.data_loader()

result = []
for i in test_sms_words:
    result.append(model.predict(i))

print('---------------------- confusion matrix\n',confusion_matrix(test_labels,result))

print('---------------------- report\n',classification_report(test_labels,result))


"""
---------------------- confusion matrix
 [[83  2]
 [ 1 14]]
---------------------- report
               precision    recall  f1-score   support

           0       0.99      0.98      0.98        85
           1       0.88      0.93      0.90        15

    accuracy                           0.97       100
   macro avg       0.93      0.95      0.94       100
weighted avg       0.97      0.97      0.97       100
"""