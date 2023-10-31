# 일반적인 반복문 + 리스트
result = []

for i in range(10):
    result.append(i)

print(result)

# 리스트 컴프리헨션
result = [i for i in range(10)]
print(result)

# 일반적인 반복문 + 리스트
result = []

for i in range(10):
    if i % 2 == 0:
        result.append(i)

print(result)

# 리스트 컴프리헨션
result = [i for i in range(10) if i % 2 == 0]
print(result)