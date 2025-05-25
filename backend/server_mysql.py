import sys
import os
import mysql.connector
from mysql.connector import Error
import json
import re
from typing import Dict, Union, List
from mcp.server.fastmcp import FastMCP, mcp

# Initialize MCP Server
mcp_server = FastMCP(name="MySQLChatServer")

# --- Database Configuration ---
# Replace with your actual database credentials
# Consider using environment variables or a config file for security
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''), # Default to empty password if not set
    'database': os.getenv('DB_NAME', 'sample_db') # Use 'sample_db' if not set
}

# --- Database Connection Helper ---
def create_connection():
    """Creates and returns a MySQL database connection."""
    connection = None
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            # print(f"Connected to MySQL Database: {db_config['database']}", file=sys.stderr)
            return connection
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}", file=sys.stderr)
        # Return a dictionary indicating the error, as the tool expects JSON-like output
        return {"error": f"Database connection failed: {e}"}
    # Return None or error dict if connection failed before is_connected check
    if connection is None:
         return {"error": "Failed to establish database connection."}
    return connection # Should not be reached if connection failed

# --- Safe Query Execution ---
def execute_select_query(connection, query: str) -> Union[Dict, str]:
    """Executes a SELECT query safely and returns results."""
    if isinstance(connection, dict) and "error" in connection:
        # Return connection error if present, already formatted as JSON/Dict
        # Convert dict to JSON string for consistent return type
        return json.dumps(connection)

    if not connection or not connection.is_connected():
         return json.dumps({"error": "Database connection is not available."})

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        # Basic check to ensure the query received is likely safe, although is_safe_query did the main check
        if not query.lower().strip().startswith("select"):
            raise ValueError("execute_select_query should only receive SELECT statements.")
        cursor.execute(query)
        result = cursor.fetchall()

        if cursor.description is None: # Handle queries that don't return rows (e.g., SHOW WARNINGS)
             # Although is_safe_query prevents DDL/DML, some SELECTs might not return rows/columns
             return json.dumps({"message": "Query executed, but no standard result set was returned."})

        column_names = [col[0] for col in cursor.description]
        # Ensure results are JSON serializable (dates, decimals etc. might need conversion)
        serializable_rows = []
        for row in result:
            serializable_row = {}
            for key, value in row.items():
                # Convert common non-serializable types
                if isinstance(value, (bytes)):
                     serializable_row[key] = value.decode('utf-8', errors='replace') # Handle potential decoding errors
                elif hasattr(value, 'isoformat'): # Handles date, datetime, time
                    serializable_row[key] = value.isoformat()
                elif isinstance(value, (Decimal)): # Handle Decimal type
                    serializable_row[key] = float(value) # Or convert to string str(value) if precision is critical
                # Add other type conversions if needed (e.g., timedelta)
                else:
                    serializable_row[key] = value
            serializable_rows.append(serializable_row)

        return json.dumps({"columns": column_names, "rows": serializable_rows})

    except Error as e:
        return json.dumps({"error": f"Database query failed: {e}"})
    except ValueError as ve: # Catch the ValueError raised above
        return json.dumps({"error": str(ve)})
    finally:
        if cursor:
            cursor.close()
        # Close the connection here as it's opened per query execution
        if connection and not (isinstance(connection, dict)) and connection.is_connected():
            connection.close()
            # print("Database connection closed after query execution.", file=sys.stderr)


# --- Query Safety Check ---
def is_safe_query(query: str) -> bool:
    """Checks if the query is a read-only SELECT query."""
    query_lower = query.lower().strip()
    # Check if it starts with 'select'
    # Allowing SHOW and DESCRIBE might be useful for schema introspection by the LLM, but stick to SELECT for now.
    if not query_lower.startswith("select "):
        print(f"Query rejected: Does not start with SELECT.", file=sys.stderr)
        return False

    # Check for disallowed keywords that modify data or structure
    # Be cautious and block potentially harmful operations.
    unsafe_keywords = [
        "insert", "update", "delete", "drop", "truncate", "alter",
        "create", "grant", "revoke", "commit", "rollback", "set",
        "lock", "unlock", "rename", "call", "do", "handler", # Added handler
        ";" # Basic check for multiple statements
    ]
    # Use word boundaries to avoid partial matches (e.g., 'selection' containing 'select')
    for keyword in unsafe_keywords:
         # Check for keyword possibly surrounded by spaces, parentheses, or at the end/start of the string
         if re.search(r'(?:^|[\s(,;])' + keyword + r'(?:$|[\s),;])', query_lower):
              print(f"Query rejected: Contains disallowed keyword '{keyword}'.", file=sys.stderr)
              return False

    # Check for comments that might hide disallowed keywords
    if '--' in query or '/*' in query:
         print(f"Query rejected: Contains SQL comments which might obscure intent.", file=sys.stderr)
         return False # Disallow comments for added safety

    print(f"Query accepted as safe: {query}", file=sys.stderr)
    return True

# --- MCP Tool: Execute SQL Query ---
@mcp.tool()
def execute_sql_query_tool(query: str) -> str:
    """
    Executes a read-only SQL SELECT query against the database and returns the results as a JSON string.
    IMPORTANT: Only SELECT queries are allowed. Do not attempt INSERT, UPDATE, DELETE, or other modifying queries.

    Args:
        query (str): The SQL SELECT query to execute.

    Returns:
        str: A JSON string containing the query results (columns and rows)
             or an error message (also as a JSON string). Example success: '{"columns": ["col1"], "rows": [{"col1": "value1"}]}'. Example error: '{"error": "..."}'.
    """
    print(f"Received query for execution: {query}", file=sys.stderr)
    if not is_safe_query(query):
        return json.dumps({"error": "Query rejected for safety reasons. Only read-only SELECT queries without comments or modifying keywords are allowed."})

    connection = create_connection()
    # create_connection now returns an error dict if failed
    if isinstance(connection, dict) and "error" in connection:
        # Connection error already contains 'error' key, just need to dump it
        return json.dumps(connection)

    # No 'finally connection.close()' here - execute_select_query handles connection closing now
    result_json = execute_select_query(connection, query)
    return result_json


# --- MCP Resource: List Tables ---
def get_table_names(connection) -> Union[List[str], Dict]:
    """Fetches table names. Returns list or error dict."""
    if isinstance(connection, dict) and "error" in connection:
        return connection # Pass connection error through

    if not connection or not connection.is_connected():
         return {"error": "Database connection is not available for getting tables."}

    cursor = None
    try:
        cursor = connection.cursor()
        # Use information_schema for potentially better compatibility/standardization
        # cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()") # Alternative
        cursor.execute("SHOW TABLES")
        # Ensure table names are strings
        tables = [str(row[0]) for row in cursor.fetchall()]
        print(f"Fetched table names: {tables}", file=sys.stderr)
        return tables
    except Error as e:
        print(f"Error fetching table names: {e}", file=sys.stderr)
        return {"error": f"Failed to fetch table names: {e}"}
    finally:
        if cursor:
             cursor.close()
        # Close connection here
        if connection and not (isinstance(connection, dict)) and connection.is_connected():
            connection.close()
            # print("Database connection closed after getting tables.", file=sys.stderr)


@mcp.resource("tables")
def get_tables() -> str:
    """
    Provides a list of table names in the database as a JSON string.
    Returns:
        str: A JSON string containing a list of table names, e.g., '["table1", "table2"]', or an error object '{"error": "..."}'.
    """
    connection = create_connection()
    # No 'finally connection.close()' here - get_table_names handles connection closing
    result = get_table_names(connection)
    # Ensure result (list or error dict) is returned as JSON string
    return json.dumps(result)


# --- MCP Resource: Get Table Schema ---
def get_table_schema(connection, table_name: str) -> Union[List[Dict], Dict]:
    """Fetches schema for a table. Returns list of column dicts or error dict."""
    if isinstance(connection, dict) and "error" in connection:
        return connection # Pass connection error through

    if not connection or not connection.is_connected():
        return {"error": "Database connection is not available for getting schema."}

    # Validate table_name format to prevent SQL injection in DESCRIBE/SHOW COLUMNS
    if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
         print(f"Invalid table name format rejected: {table_name}", file=sys.stderr)
         return {"error": f"Invalid table name format: {table_name}"}

    cursor = None
    try:
        cursor = connection.cursor(dictionary=True)
        # Using SHOW COLUMNS FROM `` can be an alternative to DESCRIBE ``
        # cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
        cursor.execute(f"DESCRIBE `{table_name}`") # Use backticks for safety
        schema = cursor.fetchall()
        print(f"Fetched schema for table '{table_name}': {len(schema)} columns", file=sys.stderr)
        # Schema from dictionary cursor should be serializable directly
        return schema
    except Error as e:
        print(f"Error describing table {table_name}: {e}", file=sys.stderr)
        # Provide a more specific error if table doesn't exist
        if 'mysql.connector.errors' in sys.modules: # Check if error codes module is available
              if e.errno == mysql.connector.errorcode.ER_NO_SUCH_TABLE:
                   return {"error": f"Table '{table_name}' not found."}
        # Generic error if specific check fails or error code unknown
        return {"error": f"Failed to describe table '{table_name}': {e}"}
    finally:
        if cursor:
            cursor.close()
        # Close connection here
        if connection and not (isinstance(connection, dict)) and connection.is_connected():
            connection.close()
            # print(f"Database connection closed after getting schema for {table_name}.", file=sys.stderr)


@mcp.resource("tables/{table_name}/schema")
def get_table_schema_resource(table_name: str) -> str:
    """
    Provides the schema (columns, types, etc.) for a specific table
    as a JSON string.

    Args:
        table_name (str): The name of the table.

    Returns:
        str: A JSON string containing the table schema (list of column descriptions)
             or an error message object '{"error": "..."}'.
             Example: '[{"Field": "id", "Type": "int", ...}, {"Field": "name", "Type": "varchar(255)", ...}]'
    """
    print(f"Request received for schema of table: {table_name}", file=sys.stderr)
    connection = create_connection()
    # No 'finally connection.close()' here - get_table_schema handles connection closing
    schema_result = get_table_schema(connection, table_name)
    # Ensure result (list or error dict) is returned as JSON string
    # Add handling for Decimal if it appears in schema descriptions (less likely)
    try:
        return json.dumps(schema_result)
    except TypeError as te:
        print(f"Serialization error for schema of {table_name}: {te}", file=sys.stderr)
        # Attempt a basic string conversion if direct serialization fails
        try:
            return json.dumps([str(item) for item in schema_result])
        except Exception:
            return json.dumps({"error": f"Could not serialize schema for table '{table_name}'."})


# --- Main Execution Block ---
if __name__ == "__main__":
    # Import Decimal here if needed for type checking, prevents import error if not installed
    from decimal import Decimal
    print("Starting MCP server for MySQL database interaction...", file=sys.stderr)
    # Run the server using standard I/O transport
    mcp_server.run(transport="stdio")
    print("MCP server stopped.", file=sys.stderr)