from torchtext.data import Field,TabularDataset,BucketIterator

tokenize = lambda x:x.split()

quote = Field(sequential=True,use_vocab=True,tokenize=tokenize,lower=True)

score = Field(sequential=False,use_vocab=False)

fields = {'quote':('q',quote),'score':('s',score)}

train_data, test_data = TabularDataset.splits(
        path="",
        train="train.json",
        test="test.json",
        format='json',
        fields=fields
        )


print(train_data[0].__dict__.keys())
print(train_data[0].__dict__.values())


quote.build_vocab(train_data,max_size=10000,min_freq=1)

train_iterator, test_iterator = BucketIterator.splits(
        (train_data,test_data),
        batch_size=2,
        device="cpu"
        )


for batch in train_iterator:
    print(batch.q)

