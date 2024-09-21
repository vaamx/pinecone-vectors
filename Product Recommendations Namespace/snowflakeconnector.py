import snowflake.connector

# Establish connection
conn = snowflake.connector.connect(
    user='OPSCALEAI',
    password='Opscale2030',
    account='nvvmnod-mw08757',
    warehouse='DIANA_DATA_LAKE',
    database='DIANA_SALES_ES',
    schema='SALES'
)
