class Person:
    def __init__(self, id, name, folder):
        self.id = id
        self.name = name
        self.folder = folder

    def __repr__(self):
        return f"Person(ID={self.id}, Name='{self.name}', Folder='{self.folder}')"
