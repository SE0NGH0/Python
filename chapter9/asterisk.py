# 가변 인수
def asterisk_test(a, *args):
    print(a, args)
    print(type(args))

print(asterisk_test(1, 2, 3, 4, 5, 6))

# 키워드 가변 인수
def asterisk_test(a, **kargs):
    print(a, kargs)
    print(type(kargs))

print(asterisk_test(1, b=2, c=3, d=4, e=5, f=6))

# 별표의 언패킹 기능
def asterisk_test(a, args):
    print(a, *args)
    print(type(args))

print(asterisk_test(1, (2, 3, 4, 5, 6)))

def asterisk_test(a, *args):
    print(a, args)
    print(type(args))

print(asterisk_test(1, *(2, 3, 4, 5, 6)))

a, b, c = ([1, 2], [3, 4], [5, 6])
print(a, b, c)

data = ([1, 2], [3, 4], [5, 6])
print(*data)

for data in zip(*[[1, 2], [3, 4], [5, 6]]):
    print(data)
    print(type(data))

def asterisk_test(a, b, c, d,):
    print(a, b, c, d)

data = {"b":1, "c":2, "d":3}
print(asterisk_test(10, **data))