-- Run only after image upload and approval of ongoing Snowflake credit use.
USE ROLE EWASTE_DEPLOYER;
USE DATABASE EWASTE_ML;
USE SCHEMA APP;
CREATE SERVICE EWASTE_API
  IN COMPUTE POOL EWASTE_API_POOL
  FROM SPECIFICATION $$
spec:
  containers:
    - name: api
      image: /ewaste_ml/app/images/ewaste-api:v1
      resources:
        requests:
          cpu: 1
          memory: 2Gi
        limits:
          memory: 4Gi
      readinessProbe:
        port: 8080
        path: /api/health
  endpoints:
    - name: api
      port: 8080
      public: true
      protocol: HTTP
serviceRoles:
  - name: api_caller
    endpoints:
      - api
$$
  MIN_INSTANCES = 1
  MAX_INSTANCES = 1;
GRANT SERVICE ROLE EWASTE_API!api_caller TO ROLE EWASTE_API_CLIENT;
SHOW SERVICE CONTAINERS IN SERVICE EWASTE_API;
SHOW ENDPOINTS IN SERVICE EWASTE_API;
-- After testing, suspend to stop the running workload:
-- ALTER SERVICE EWASTE_API SUSPEND;
-- Pool auto-suspend applies once no workloads remain; it does not stop an active API.
-- To restart:
-- ALTER SERVICE EWASTE_API RESUME;
