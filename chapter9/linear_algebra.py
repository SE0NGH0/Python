# 벡터의 연산
u = [2, 2]
v = [2, 3]
z = [3, 5]
result = []

for i in range(len(u)):
    result.append(u[i] + v[i] + z[i])

print(result)

result = [sum(t) for t in zip(u, v, z)]
print(result)

print([t for t in zip(u, v, z)])

# 별표를 사용한 함수화
def vector_addition(*args):
    return [sum(t) for t in zip(*args)]

print(vector_addition(u, v, z))

row_vectors = [[2, 2], [2, 3], [3, 5]]
print(vector_addition(*row_vectors))

# 스칼라 - 벡터 연산
u = [1, 2, 3]
v = [4, 4, 4]
alpha = 2

result = [alpha * sum(t) for t in zip(u, v)]
print(result)

# 행렬의 연산
matrix_a = [[3, 6], [4, 5]]
matrix_b = [[5, 8], [6, 7]]
result = [[sum(row) for row in zip(*t)] for t in zip(matrix_a, matrix_b)]
print(result)

print([t for t in zip(matrix_a, matrix_b)])

# 행렬의 동치
matrix_a = [[1, 1], [1, 1]]
matrix_b = [[1, 1], [1, 1]]
print(all([row[0] == value for t in zip(matrix_a, matrix_b) for row in zip(*t) for value in row]))

matrix_b = [[5, 8], [6, 7]]
print(all([all([row[0] == value for value in row]) for t in zip(matrix_a, matrix_b) for row in zip(*t)]))

print(any([False, False, False]))
print(any([False, True, False]))
print(all([False, True, True]))
print(all([True, True, True]))

print([[row[0] == value for value in row] for t in zip(matrix_a, matrix_b) for row in zip(*t)])

matrix_a = [[1, 2, 3], [4, 5, 6]]
result = [[element for element in t] for t in zip(*matrix_a)]
print(result)

# 행렬의 곱셈
matrix_a = [[1, 1, 2], [2, 1, 1]]
matrix_b = [[1, 1], [2, 1], [1, 3]]
result = [[sum(a * b for a, b in zip(row_a, column_b)) for column_b in zip(*matrix_b)] for row_a in matrix_a]
print(result)