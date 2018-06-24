import codecs

def get_data(filename):
	tweets = []
	with codecs.open('./'+filename, 'r', encoding='utf-8') as f:
		data = f.readlines()
	for line in data:
		line = line
		a = line.split()
		c = a[-1]
		a.pop()
		a = ' '.join(a)
		tweets.append({
                'text': a,
                'label': c,
                })
	return tweets


