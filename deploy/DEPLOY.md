# Deploying Waste Sorter to AWS Lambda

The app runs as a container on AWS Lambda behind a public function URL. It costs nothing when idle. The first request after
a quiet spell is slow (a cold start of several seconds while TensorFlow and the
model load); requests after that are fast.

## Before you start

- AWS CLI v2, signed in with permission to create ECR repos, IAM roles and Lambda functions
- Docker running
- A trained model in `models/`: `waste_classifier.keras`, `.labels.json` and `.metrics.json`
  (run `python -m waste_classifier.train --arch mobilenet --epochs 5`)

## First deploy

```bash
export AWS_REGION=eu-west-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=waste-sorter
export FUNCTION_NAME=waste-sorter
export IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:lambda"

# 1. Container registry
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"

# 2. Build natively for arm64 (don't force linux/amd64 on Apple Silicon), then push.
#    --provenance/--sbom=false: newer Docker adds attestation data that Lambda rejects
#    ("image manifest ... media type is not supported").
docker build --provenance=false --sbom=false -f Dockerfile.lambda -t "$ECR_REPO:lambda" .
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
docker tag "$ECR_REPO:lambda" "$IMAGE_URI"
docker push "$IMAGE_URI"

# 3. Execution role: logs only, the app uses no other AWS services
aws iam create-role --role-name "${FUNCTION_NAME}-role" --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]
}'
aws iam attach-role-policy --role-name "${FUNCTION_NAME}-role" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
sleep 10   # let the new role propagate

# 4. The function. 3008 MB gives TensorFlow enough CPU; 60 s covers a cold start.
aws lambda create-function --function-name "$FUNCTION_NAME" \
  --package-type Image --code ImageUri="$IMAGE_URI" \
  --role "arn:aws:iam::${AWS_ACCOUNT_ID}:role/${FUNCTION_NAME}-role" \
  --architectures arm64 --memory-size 3008 --timeout 60 --region "$AWS_REGION"

# 5. Public HTTPS URL, no sign-in (it's a demo)
aws lambda create-function-url-config --function-name "$FUNCTION_NAME" --auth-type NONE --region "$AWS_REGION"
aws lambda add-permission --function-name "$FUNCTION_NAME" \
  --statement-id FunctionURLAllowPublicAccess --action lambda:InvokeFunctionUrl \
  --principal "*" --function-url-auth-type NONE --region "$AWS_REGION"
aws lambda add-permission --function-name "$FUNCTION_NAME" \
  --statement-id FunctionURLAllowInvokeAction --action lambda:InvokeFunction \
  --principal "*" --invoked-via-function-url --region "$AWS_REGION"

aws lambda get-function-url-config --function-name "$FUNCTION_NAME" \
  --region "$AWS_REGION" --query FunctionUrl --output text
```

## Redeploy after retraining or code changes

```bash
docker build --provenance=false --sbom=false -f Dockerfile.lambda -t "$ECR_REPO:lambda" .
docker tag "$ECR_REPO:lambda" "$IMAGE_URI"
docker push "$IMAGE_URI"
aws lambda update-function-code --function-name "$FUNCTION_NAME" --image-uri "$IMAGE_URI" --region "$AWS_REGION"
```

## Test the image locally first

The Lambda base image includes AWS's runtime emulator:

```bash
docker run --rm -p 9000:8080 waste-sorter:lambda
curl -s localhost:9000/2015-03-31/functions/function/invocations \
  -d '{"version":"2.0","rawPath":"/api/status","requestContext":{"http":{"method":"GET","path":"/api/status","sourceIp":"127.0.0.1"}},"headers":{},"isBase64Encoded":false}'
```

## Things to know

- **Upload size.** Function URLs cap requests at about 6 MB. The page shrinks photos
  in the browser before uploading, so phone photos are fine.
- **Cost.** About $0.25/month to store the image in ECR, plus fractions of a cent per request.
