from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("app_request_count", "Total HTTP requests", ["method", "endpoint", "http_status"])
PREDICTION_COUNT = Counter("app_prediction_count", "Number of predictions", ["status"])
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["endpoint"]) 
