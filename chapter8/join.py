colors = ['red', 'blue', 'green', 'yellow']
result = ''.join(colors)    # join()을 활용하여 리스트의 각 요소를 빈칸 없이 연결
print(result)

result = ''.join(colors)    # 연결 시, 1칸을 띄고 연결
print(result)

result = ', '.join(colors)  # 연결 시 ","로 연결
print(result)

result = '-'.join(colors)   # 연결 시 "-"로 연결
print(result)