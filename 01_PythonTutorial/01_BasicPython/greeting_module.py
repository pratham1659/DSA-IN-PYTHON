print("Module is being loaded / executed")

def greet():
    print("Hello from greeting_module!")

if __name__ == "__main__":
    print("Running directly! __name__ =", __name__)
    greet()
else:
    print("Imported as a module! __name__ =", __name__)
