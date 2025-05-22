from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Replace with your actual MongoDB connection string
# IMPORTANT: It's recommended to use environment variables for sensitive data like URIs
uri = "mongodb+srv://SweetAgent:gOiIvsNMFd3GkY9R@sweetagent.92eclr4.mongodb.net/?retryWrites=true&w=majority&appName=SweetAgent" # This URI is for demonstration and will be ******** in logs.

DB_NAME = "nba_stats_db"

def get_mongo_db():
    """
    Connects to MongoDB and returns the database object.
    """
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return None
        
    db = client[DB_NAME]
    return db

# For direct execution or testing connection
if __name__ == "__main__":
    db_object = get_mongo_db()
    if db_object:
        print(f"Successfully obtained database object for '{DB_NAME}'.")
        print(f"Collections in the database: {db_object.list_collection_names()}")
    else:
        print(f"Failed to obtain database object for '{DB_NAME}'.")

# Expose the db object for import if this module is imported elsewhere,
# though it's generally better to call get_mongo_db() explicitly.
# For this task, dataextract25.py will import and call get_mongo_db().
# db = get_mongo_db() # Avoid top-level call to get_mongo_db to prevent connection on import.
