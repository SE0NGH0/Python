from collections import deque

deque_list = deque()
deque_list2 = deque()

for i in range(5):
    deque_list.append(i)

print(deque_list)

deque_list.pop()
print(deque_list)

deque_list.pop()
print(deque_list)

deque_list.pop()
print(deque_list)

for i in range(5):
    deque_list2.appendleft(i)

print(deque_list2)

print(deque(reversed(deque_list2)))

deque_list2.rotate(2)
print(deque_list2)

deque_list2.rotate(2)
print(deque_list2)

deque_list2.extend([5, 6, 7])
print(deque_list2)

deque_list2.extendleft([5, 6, 7])
print(deque_list2)
