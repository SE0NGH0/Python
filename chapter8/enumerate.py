for i, v in enumerate(['tic', 'tac', 'toe']):   # 리스트에 있는 인덱스와 값을 언패킹
    print(i, v)

print({i:j for i, j in enumerate('PYTHON is an language of programming.'.split())})

alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']

for a, b in zip(alist, blist):
    print(a, b) # 병렬로 값을 추출

a, b, c = zip((1, 2, 3), (10, 20, 30), (100, 200, 300))
print(a, b, c)

print([sum(x) for x in zip((1, 2, 3), (10, 20, 30), (100, 200, 300))])

alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']

for i, (a, b) in enumerate(zip(alist, blist)):
    print(i, a, b)  # (인덱스, alist[인덱스], blist[인덱스]) 표시
    