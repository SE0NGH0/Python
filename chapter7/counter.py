from collections import Counter

text = list("gallahad")
print(text)

c = Counter(text)
print(c)

print(c["a"])

c = Counter({'red': 4, 'blue': 2})
print(c)
print(list(c.elements()))

c = Counter(cats = 4, dogs = 8)
print(c)
print(list(c.elements()))

c = Counter(a = 4, b = 2, c = 0, d = -2)
d = Counter(a = 1, b = 2, c = 3, d = 4)
c.subtract(d)   # c - d
print(c)

c = Counter(a = 4, b = 2, c = 0, d = -2)
d = Counter(a = 1, b = 2, c = 3, d = 4)
print(c + d)
print(c & d)    # 두 객체에 같은 값이 있을 때
print(c | d)    # 두 객체에서 하나가 포함되어 있다면, 그리고 좀 더 큰 값이 있다면 그 값으로 합집합 적용

from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(11, y=22)
print(p)

print(p.x, p.y)
print(p[0] + p[1])
