words = 'The quick brown fox jumps over the lazy dog'.split()
print(words)

stuff = [[w.upper(), w.lower(), len(w)] for w in words] # 리스트의 각 요소를 대문자, 소문자, 길이로 변환하여 이차원 리스트로 변환

for i in stuff:
    print(i)

case_1 = ["A", "B", "C"]
case_2 = ["D", "E", "A"]
result = [i + j for i in case_1 for j in case_2]
print(result)

result = [[i + j for i in case_1] for j in case_2]
print(result)