import os
import csv
from arango import ArangoClient

# Connect to ArangoDB
client = ArangoClient(hosts="http://localhost:8529")
db = client.db("arxiv_repository", username="root", password=os.environ.get("ARANGO_PASSWORD"))

csv_filename = "missing_embeddings.csv"
print("Creating CSV of missing embeddings...")

# Use streaming cursor for better performance
query = """
FOR doc IN arxiv_metadata
    LET emb = DOCUMENT("arxiv_abstract_embeddings", CONCAT(doc._key, "_chunk_0_emb"))
    FILTER emb == null
    RETURN {
        arxiv_id: doc._key,
        submission_date: doc.update_date,
        abstract_length: LENGTH(doc.abstract)
    }
"""

# Open CSV and write header
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["arxiv_missing_embedding", "submission_date", "abstract_length"])
    
    # Create cursor with batch size
    cursor = db.aql.execute(
        query,
        batch_size=100,
        stream=True
    )
    
    count = 0
    for doc in cursor:
        writer.writerow([
            doc["arxiv_id"] if doc["arxiv_id"] else "",
            doc["submission_date"] if doc["submission_date"] else "",
            doc["abstract_length"] if doc["abstract_length"] is not None else 0
        ])
        count += 1
        if count % 500 == 0:
            print(f"Processed {count} records...")
    
    cursor.close()

print(f"\nTotal records: {count}")
print(f"CSV saved as: {csv_filename}")

# Show sample
print("\nFirst 5 entries:")
with open(csv_filename, "r") as f:
    for i, line in enumerate(f):
        if i < 6:
            print(line.strip())
