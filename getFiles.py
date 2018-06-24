m={}
for line in open('emolex.txt'):
	line = line.strip().split('\t')
	if line[2]=='1':
		if line[1] not in m:
			m[line[1]] = [line[0]]
		else:
			m[line[1]].append(line[0])

for cat in m:
	f = open('./lexicons/'+cat+'.txt','w')
	for word in m[cat]:
		f.write(word+'\n')