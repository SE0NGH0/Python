word_1 = "Hello"
word_2 = "World"
result = [i + j for i in word_1 for j in word_2]    # 중첩 반복문
print(result)

case_1 = ["A", "B", "C"]
case_2 = ["D", "E", "A"]
result = [i + j for i in case_1 for j in case_2 if not(i==j)]
print(result)