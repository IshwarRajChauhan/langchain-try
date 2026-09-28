from pydantic import BaseModel



class person(BaseModel):
    name: str
    age: int
    city: str

person1 = person(name="John", age=30, city="Jeher")
print(person1)


