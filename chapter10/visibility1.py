class Product(object):
    pass

class Inventory(object):
    def __init__(self):
        self.__items = []   

    def add_new_item(self, product):
        if type(product) == Product:
            self.__items.append(product)
            print("new item added")

        else:
            raise ValueError("Invalid Item")
        
    def get_number_of_items(self):
        return len(self.__items)

    def __init__(self):
        self.__items = []   # private 변수로 선언 (타인이 접근 못 함)

    @property       # property 데코레이터 (숨겨진 변수 반환)
    def items(self):
        return self.__items
    
my_inventory = Inventory()
my_inventory.add_new_item(Product())
my_inventory.add_new_item(Product())

my_inventory.__items