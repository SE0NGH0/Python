# 일반적인 반복문 + 리스트
def scalar_vector_product(scalar, vector):
    result = []
    for value in vector:
        result.append(scalar * value)
    return result

iteration_max = 10000

vector = list(range(iteration_max))
scalar = 2

for _ in range(iteration_max):
    scalar_vector_product(scalar, vector)

# 리스트 컴프리헨션
iteration_max = 10000

vector = list(range(iteration_max))
scalar = 2

for _ in range(iteration_max):
    [scalar * value for value in range(iteration_max)]