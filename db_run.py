import sqlite3
DATABASE_PATH = "sales_data.db"

def setup_sample_database():
    """Create sample database with sales data"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        city TEXT,
        country TEXT,
        registration_date DATE
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT NOT NULL,
        category TEXT,
        price DECIMAL(10,2),
        stock_quantity INTEGER
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        order_date DATE,
        total_amount DECIMAL(10,2),
        status TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        unit_price DECIMAL(10,2),
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    ''')
    
    # Insert sample data
    customers_data = [
        (1, 'John Doe', 'john@email.com', 'Jakarta', 'Indonesia', '2024-01-15'),
        (2, 'Jane Smith', 'jane@email.com', 'Surabaya', 'Indonesia', '2024-02-20'),
        (3, 'Bob Johnson', 'bob@email.com', 'Bandung', 'Indonesia', '2024-03-10'),
        (4, 'Alice Brown', 'alice@email.com', 'Medan', 'Indonesia', '2024-04-05'),
        (5, 'Charlie Wilson', 'charlie@email.com', 'Yogyakarta', 'Indonesia', '2024-05-12'),
        (6, 'David Lee', 'david@email.com', 'Semarang', 'Indonesia', '2024-06-15'),
        (7, 'Evelyn Martin', 'evelyn@email.com', 'Medan', 'Indonesia', '2024-07-20'),
        (8, 'Florence Brown', 'florence@email.com', 'Bali', 'Indonesia', '2024-08-10'),
        (9, 'Gabriel Davis', 'gabriel@email.com', 'Medan', 'Indonesia', '2024-09-05'),
        (10, 'Hannah Johnson', 'hannah@email.com', 'Lampung', 'Indonesia', '2024-10-12')
    ]
    
    products_data = [
        (1, 'Laptop Dell', 'Electronics', 15000000, 50),
        (2, 'iPhone 15', 'Electronics', 18000000, 30),
        (3, 'Nike Shoes', 'Fashion', 2500000, 100),
        (4, 'Coffee Maker', 'Home Appliances', 1200000, 25),
        (5, 'Book - Python Programming', 'Books', 350000, 200) , 
        (6, 'Book - Data Science', 'Books', 400000, 100)
    ]
    
    orders_data = [
        (1, 1, '2024-01-20', 15000000, 'Completed'),
        (2, 2, '2024-02-25', 18000000, 'Completed'),
        (3, 3, '2024-03-15', 2500000, 'Completed'),
        (4, 1, '2024-04-10', 1550000, 'Pending'),
        (5, 4, '2024-05-15', 2850000, 'Completed')
    ]
    
    order_items_data = [
        (1, 1, 1, 1, 15000000),  # John bought 1 Laptop
        (2, 2, 2, 1, 18000000),  # Jane bought 1 iPhone
        (3, 3, 3, 1, 2500000),   # Bob bought 1 Nike Shoes
        (4, 4, 4, 1, 1200000),   # John bought 1 Coffee Maker
        (5, 4, 5, 1, 350000),    # John bought 1 Book
        (6, 5, 3, 1, 2500000),   # Alice bought 1 Nike Shoes
        (7, 5, 5, 1, 350000)     # Alice bought 1 Book
    ]
    
    cursor.executemany('INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?)', customers_data)
    cursor.executemany('INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)', products_data)
    cursor.executemany('INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?)', orders_data)
    cursor.executemany('INSERT OR REPLACE INTO order_items VALUES (?, ?, ?, ?, ?)', order_items_data)
    
    conn.commit()
    conn.close()
    print("Sample database created successfully!")
    
setup_sample_database()