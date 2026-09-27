# Snowflake API deployment

Status: configuration prepared; not deployed or runtime-tested.
Target: Snowpark Container Services, linux/amd64, one service instance.
Existing API paths are preserved. Camera capture happens on the client device;
the website/app uploads to its backend, which authenticates to this service.

## Account prerequisites

Use a Snowflake account with Snowpark Container Services support in its region.
Snowflake's tutorial prerequisites explicitly exclude trial accounts. Confirm
eligibility with your account administrator before starting.
An administrator must run 01_setup.sql and grant EWASTE_DEPLOYER to the
deployment user. Review the new object names for conflicts before running.
No existing warehouse, customer tables, or datasets need modification.
The API uses bundled models and CSV data; it does not need a query warehouse.

Compute pools consume credits while workloads run. Image storage and data
transfer can also be billed. Check the account's regional rates and budget.
The proposed pool is capped at one CPU_X64_XS node and starts suspended.
Starting the service resumes the pool. Pool auto-suspend is NOT a spending cap:
a running API keeps consuming compute even when no users are sending requests.
After a test, suspend the service using the command in 02_service.sql.

## Build and smoke-test on a machine with Docker

From the repository root, run:

```bash
docker build --platform linux/amd64 -f deploy/snowflake/Dockerfile -t ewaste-api:v1 .
docker run --rm -p 8080:8080 --name ewaste-api-test ewaste-api:v1
```

In another terminal, test /api/health, upload a known image to /api/predict,
then request /api/recommend-price with the resulting category and item details.
See ../../DEPLOYMENT.md for the exact request and response fields.
Measure memory and verify v5 preprocessing against known correct predictions.
The container build checks classifier SHA256, restores the pricing bundle,
and imports both engines. A successful build is not an accuracy test.

## Upload the image

After 01_setup.sql, copy repository_url from SHOW IMAGE REPOSITORIES.
It resembles org-account.registry.snowflakecomputing.com/ewaste_ml/app/images.
Authenticate with Snowflake CLI using your account's approved method, then:

```bash
snow spcs image-registry login --connection YOUR_CONNECTION
docker tag ewaste-api:v1 YOUR_REPOSITORY_URL/ewaste-api:v1
docker push YOUR_REPOSITORY_URL/ewaste-api:v1
```

Replace the capitalized placeholders; they are not real credentials or URLs.
Never commit credentials. Snowflake CLI and Docker must be installed.
Keep the image tag in 02_service.sql synchronized with the uploaded tag.
Use a new version tag for future releases rather than overwriting a deployed tag.

## Start and verify

Run 02_service.sql as EWASTE_DEPLOYER after approving credit consumption.
Check SHOW SERVICE CONTAINERS reports a ready container. Get the actual
ingress_url from SHOW ENDPOINTS; do not construct or guess it.
Test authenticated health, known-image classification, and a pricing quote.
Review service logs if startup fails. This environment has not built the image
or executed either SQL file; dependency and account compatibility remain gates.

## Backend integration and authentication

A Snowflake public endpoint still requires Snowflake authentication.
An admin should grant EWASTE_API_CLIENT to a dedicated backend identity.
That identity needs an account-approved PAT or key-pair authentication setup.
Follow Snowflake's programmatic endpoint tutorial for the token exchange and
required request headers. PAT authentication can use the Authorization header
in the form Snowflake Token="<PAT>". Do not log that header.

Keep the credential only in the developer's backend secret store. The public
website/mobile app calls the developer's backend, which forwards image uploads
and pricing requests to Snowflake. Do not embed a Snowflake token in frontend
JavaScript or a distributed mobile app.

The 2 GiB request/4 GiB limit is an initial configuration, not a measured
requirement. Verify actual memory and latency before production use.

## Official references

- https://docs.snowflake.com/en/developer-guide/snowpark-container-services/tutorials/common-setup
- https://docs.snowflake.com/en/developer-guide/snowpark-container-services/specification-reference
- https://docs.snowflake.com/en/developer-guide/snowpark-container-services/tutorials/advanced/tutorial-8-access-public-endpoint-programmatically
- https://docs.snowflake.com/en/developer-guide/snowpark-container-services/accounts-orgs-usage-views
