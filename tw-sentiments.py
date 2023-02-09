from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

#preprocess tweets 
tweet='''@VirginAmerica @virginmedia I'm flying your #fabulous #Seductive skies again! U take all the #stress away from travel http://t.co/ahlXHhKiyn'''

tweet_words=[]

for word in tweet.split(' '): # splitting based on space
    if word.startswith('@') and len(word)>1:
        word= '@client'

    elif word.startswith('http'):
        word='http'
    tweet_words.append(word)
tweet_processed=(' ').join(tweet_words)

#print(tweet_processed)

#loading model and tokenizer
roberta= "cardiffnlp/twitter-roberta-base-sentiment"

model= AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer=AutoTokenizer.from_pretrained(roberta)
lables=['Negative', 'Neutral', 'Positive']

#sentiment analysis

encoded_tweet= tokenizer(tweet_processed, return_tensors='pt')
#print(encoded_tweet)
#output=model(encoded_tweet['input_ids'],encoded_tweet['attention_mask'])
#print(output)
output=model(**encoded_tweet) # unpacked encoded tweet like above

# converting output to probablity using softmax
scores=output[0][0].detach().numpy() # only taking logits value part from output
scores=softmax(scores) # converting values to probability scale
print(scores)

for i in range(len(scores)):
    l=lables[i]
    s=scores[i]
    print(l,s)
