ex = [1, 2, 3, 4, 5]
f = lambda x: x ** 2
print(list(map(f, ex)))

# 일반적인 반복문
ex = [1, 2, 3, 4, 5]
f = lambda x: x ** 2
for value in map(f, ex):
    print(value)

# 리스트 컴프리헨션
ex = [1, 2, 3, 4, 5]
print([x ** 2 for x in ex])

# 일반적인 반복문
ex = [1, 2, 3, 4, 5]
f = lambda x, y: x + y
print(list(map(f, ex, ex)))

# 리스트 컴프리헨션
print([x + y for x, y in zip(ex, ex)])

#filtering 기능
print(list(map(lambda x: x ** 2 if x % 2 == 0 else x, ex))) # map() 함수

print([x ** 2 if x % 2 == 0 else x for x in ex])    # 리스트 컴프리헨션