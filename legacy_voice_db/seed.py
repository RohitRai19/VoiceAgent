"""Drop and recreate sample collections so the assistant works on first run."""
from datetime import datetime, timedelta

from db_tools import db

now = datetime(2026, 11, 10, 10, 0, 0)

customers = [
    {"customer_id": "C001", "name": "Rahul Sharma", "city": "Jaipur", "phone": "+91 98290 11111", "email": "rahul.sharma@example.com", "joined": now - timedelta(days=400)},
    {"customer_id": "C002", "name": "Priya Iyer", "city": "Chennai", "phone": "+91 98400 22222", "email": "priya.iyer@example.com", "joined": now - timedelta(days=320)},
    {"customer_id": "C003", "name": "Venkatesh Reddy", "city": "Hyderabad", "phone": "+91 98480 33333", "email": "venkatesh.reddy@example.com", "joined": now - timedelta(days=210)},
    {"customer_id": "C004", "name": "Ananya Banerjee", "city": "Kolkata", "phone": "+91 98300 44444", "email": "ananya.banerjee@example.com", "joined": now - timedelta(days=95)},
    {"customer_id": "C005", "name": "Mohammed Khan", "city": "Lucknow", "phone": "+91 94150 55555", "email": "mohammed.khan@example.com", "joined": now - timedelta(days=30)},
]

products = [
    {"product_id": "P001", "name": "Basmati Rice 5kg", "category": "grocery", "price": 650, "stock": 120},
    {"product_id": "P002", "name": "Cotton Kurta", "category": "clothing", "price": 1299, "stock": 45},
    {"product_id": "P003", "name": "Pressure Cooker 3L", "category": "kitchen", "price": 1850, "stock": 0},
    {"product_id": "P004", "name": "Bluetooth Earphones", "category": "electronics", "price": 2499, "stock": 18},
    {"product_id": "P005", "name": "Masala Chai 500g", "category": "grocery", "price": 320, "stock": 300},
]

orders = [
    {"order_id": "O1001", "customer_id": "C001", "customer_name": "Rahul Sharma", "product_id": "P004", "product_name": "Bluetooth Earphones", "quantity": 1, "amount": 2499, "status": "shipped", "ordered_at": now - timedelta(days=3), "expected_delivery": now + timedelta(days=2), "ship_to": "Jaipur"},
    {"order_id": "O1002", "customer_id": "C002", "customer_name": "Priya Iyer", "product_id": "P002", "product_name": "Cotton Kurta", "quantity": 2, "amount": 2598, "status": "delivered", "ordered_at": now - timedelta(days=12), "delivered_at": now - timedelta(days=8), "ship_to": "Chennai"},
    {"order_id": "O1003", "customer_id": "C003", "customer_name": "Venkatesh Reddy", "product_id": "P001", "product_name": "Basmati Rice 5kg", "quantity": 3, "amount": 1950, "status": "pending", "ordered_at": now - timedelta(days=1), "ship_to": "Hyderabad"},
    {"order_id": "O1004", "customer_id": "C004", "customer_name": "Ananya Banerjee", "product_id": "P003", "product_name": "Pressure Cooker 3L", "quantity": 1, "amount": 1850, "status": "cancelled", "ordered_at": now - timedelta(days=6), "cancel_reason": "out of stock", "ship_to": "Kolkata"},
    {"order_id": "O1005", "customer_id": "C005", "customer_name": "Mohammed Khan", "product_id": "P005", "product_name": "Masala Chai 500g", "quantity": 4, "amount": 1280, "status": "pending", "ordered_at": now - timedelta(hours=5), "ship_to": "Lucknow"},
]


def main() -> None:
    for name, docs in (("customers", customers), ("products", products), ("orders", orders)):
        db.drop_collection(name)
        db[name].insert_many(docs)
        print(f"seeded {name}: {db[name].count_documents({})} docs")
    db.customers.create_index("customer_id", unique=True)
    db.products.create_index("product_id", unique=True)
    db.orders.create_index("order_id", unique=True)
    db.orders.create_index("customer_id")


if __name__ == "__main__":
    main()
