


from torchtext.legacy.data import Field, TabluarDataset,BucketIterator

tokenize = lambda x:x.split()

quote = Field(sequential=True, use_voab=True,tokenize=tokenize,lower=True)
score = Field(sequential=False,use_vocab=False)

fields = {"quote":('q',quote), "score":('s',score)}

train_data,test_data= TabularDataset.splits(
        path='mydata',train='train.json',test='test.json',format='json',fields=fields
        )


print(train_data[0].__dict__.keys())

print(train_data[0].__dict__.values())
