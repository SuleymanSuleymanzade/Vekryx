"""Run with: python examples/basic_usage.py"""
from vekryx import Vekryx

store = Vekryx(dimensions=3)
store.insert("apple", [1.0, 2.0, 3.0])
store.insert("orange", [4.0, 5.0, 6.0])
print("Apple:", store.get("apple"))
print("Records:", len(store))

# Inserting the same ID updates its vector without adding another record.
store.insert("apple", [2.0, 2.0, 3.0])
print("Updated apple:", store.get("apple"))
print("Deleted orange:", store.delete("orange"))

try:
    store.get("orange")
except KeyError:
    print("Orange no longer exists")

try:
    store.insert("invalid", [1.0])
except ValueError as error:
    print("Invalid vector:", error)
