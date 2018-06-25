from string import punctuation
# from preprocess_twitter import tokenize as tokenizer_g
from gensim.parsing.preprocessing import STOPWORDS


def glove_tokenize(text):
    # text = tokenizer_g(text)
    text = ''.join([c for c in text if c not in '!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n'])
    words = text.split()
    # words = [word for word in words if word =='not' or word not in STOPWORDS]
    return words

def word_frame_tokenize(text,frame):
	frame = frame.split()
	tweet = text.split()
	x = []
	y = []
	#print(len(frame),len(tweet))
	for a,b in zip(tweet,frame):
		if a not in '!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n':
			x.append(a)
			y.append(b)
	#print(glove_tokenize(text))
	#print(x)
	return x,y


s = 'that i will not be at sdcc until saturday , but today i have a tour at <allcaps> uci </allcaps> ! sad'
f = 'No_frame No_frame No_frame No_frame No_frame No_frame No_frame No_frame No_frame No_frame No_frame Calendric_unit No_frame Being_obligated No_frame Travel No_frame No_frame No_frame No_frame No_frame No_frame'
word_frame_tokenize(s,f)
