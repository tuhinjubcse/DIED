from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import sys
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector

# reload(sys)
# sys.setdefaultencoding('utf8')

text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

seg_tw = Segmenter(corpus="twitter")
sp = SpellCorrector(corpus="twitter") 
f = open('train.csv','r')
f1 = open('tokenized_tweets_train.txt', 'w')
c=1
for line in f:
    a = line.strip().split('\t')
    b = a[-1]
    c = a[-2]
    b = b.split()
    for i in range(len(b)):
        if 'pic.twitter.com' in b[i]:
            b[i] = '<url>'
    b = ' '.join(b)
    a = text_processor.pre_process_doc(b)
    for i in range(len(a)):
        if a[i].isalpha():
            a[i] = seg_tw.segment(sp.correct(a[i]))
    a = ' '.join(a)
    f1.write(a+' '+c+'\n')

f = open('trial.csv','r')
l = []
for line in open('trial.labels','r'):
    l.append(line.strip())
f1 = open('tokenized_tweets_test.txt', 'w')
count =0
for line in f:
    a = line.strip().split('\t')
    b = a[-1]
    c = l[count]
    count = count+1
    b = b.split()
    for i in range(len(b)):
        if 'pic.twitter.com' in b[i]:
            b[i] = '<url>'
    b = ' '.join(b)
    a = text_processor.pre_process_doc(b)
    for i in range(len(a)):
        if a[i].isalpha():
            a[i] = seg_tw.segment(sp.correct(a[i]))
    a = ' '.join(a)
    f1.write(a+' '+c+'\n')