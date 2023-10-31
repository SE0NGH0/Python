def f():
    global s
    s = "I love Korea!"
    print(s)

s = "I love Hanguk!"
f()
print(s)