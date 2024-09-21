cur = conn.cursor()
try:
    # Execute a query
    cur.execute("SELECT PRODUCT_ID, DESCRIPTION, PRICE FROM DIANA_SALES_ES.SALES.PRODUCTS")
    rows = cur.fetchall()
finally:
    cur.close()
