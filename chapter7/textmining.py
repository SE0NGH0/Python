text = """A press release is the quickest and easiest way to get free 
publicity. If well written, a press release can result in multiple 
published artivles about your firm and its products. And that can mean new
prospects contacting you asking yout to sell to them.""".lower().split()

from collections import defaultdict

word_count = defaultdict(lambda: 0) # Default 값을 0으로 설정

for word in text:
    word_count[word] += 1

from collections import OrderedDict

for i, v in OrderedDict(sorted(word_count.items(), key=lambda t: t[1], reverse=True)).items():
    print(i, v)