class MyClass:
    x = 10

obj = MyClass()
a = hasattr(obj, 'x')        # True
b = hasattr(obj, 'y')        # False

print(a)
print(b)