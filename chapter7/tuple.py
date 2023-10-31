s = set([1, 2, 3, 1, 2, 3]) # set() 함수를 사용하여 1, 2, 3을 세트 객체로 생성
print(s)

s.add(1)    # 1을 추가하는 명령이지만 중복 불허로 추가되지 않음
print(s)

s.remove(1) # 1 삭제
print(s)

s.update([1, 4, 5, 6, 7])   # [1, 4, 5, 6, 7] 추가
print(s)

s.discard(3)    # 3 삭제
print(s)

s.clear()   # 모든 원소 삭제
print(s)

s1 = set([1, 2, 3, 4, 5])
s2 = set([3, 4, 5, 6, 7])

print(s1.union(s2))   # s1과 s2의 합집합

print(s1 | s2) # set([1, 2, 3, 4, 5, 6, 7])

print(s1.intersection(s2))  # s1과 s2의 교집합

print(s1 & s2)  # set([3, 4, 5])

print(s1.difference(s2))    # s1과 s2의 차집합

print(s1 - s2) # set([1, 2])