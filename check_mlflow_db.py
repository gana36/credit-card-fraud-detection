import sqlite3

conn = sqlite3.connect('mlflow/mlflow.db')
cursor = conn.cursor()

print("=== Experiments ===")
cursor.execute('SELECT experiment_id, name, artifact_location FROM experiments')
for row in cursor.fetchall():
    print(row)

print("\n=== Recent Runs ===")
cursor.execute('SELECT run_uuid, artifact_uri FROM runs ORDER BY start_time DESC LIMIT 5')
for row in cursor.fetchall():
    print(row)

print("\n=== Model Versions ===")
cursor.execute('SELECT version, run_id, source FROM model_versions WHERE name="credit-fraud" ORDER BY version')
for row in cursor.fetchall():
    print(row)

conn.close()
