a = ["color", 1, 0.2]
color = ['yellow', 'blue', 'green', 'black', 'purple']
a[0] = color    # 리스트 안에 리스트도 입력 가능
print(a)

b = [5, 4, 3, 2, 1]
c = [1, 2, 3, 4, 5]
c = b
print(c)

b.sort()    # 리스트에 있는 값들의 순서를 오름차순으로 변환
print(c)

d = [6, 7, 8, 9, 10]
print(b, d)