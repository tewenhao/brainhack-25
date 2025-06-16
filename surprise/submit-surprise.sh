#!/bin/bash
set -e

submit_usage() {
    echo "Usage:     ./submit-surprise.sh IMAGE-NAME:IMAGE-TAG"
    if [ -z "$TEAM_NAME" ]; then
        echo "Example:   ./submit-surprise.sh your-team-name-surprise:latest"
    else
        echo "Example:   ./submit-surprise.sh $TEAM_NAME-surprise:latest"
    fi
}

if [ -z "$TEAM_NAME" ];
    then echo "No team name found in environment! Sounds like something broke, ping @tech on Discord."
    exit 1;
fi
if [ -z "$1" ];
    then echo "No Docker image provided!"
    submit_usage
    exit 1;
fi
image_ref="$1"

if [[ "$image_ref" == *@* ]]; then
    echo "ERROR: you can't submit a digest reference!"
    submit_usage
    exit 1;
else
    image="${image_ref%%:*}"
    tag="${image_ref#*:}"
    if [[ "$image_ref" == "$image" ]]; then
        echo "WARNING: no tag given, defaulting to 'latest'."
        tag="latest"
    fi
fi

echo "Image:   $image"
echo "Tag:     ${tag:-<none>}"

if [[ "$image" == *-surprise ]]; then
    task=surprise
    port=5005
else
    echo "ERROR: could not parse task from image $image." \
        "Your image name must end with '-surprise'."
    submit_usage
    exit 1;
fi

if [[ "$image" == asia-southeast1-docker.pkg.dev/til-ai-2025/* ]]; then
    echo "Image $image is already an Artifact Registry tag, not retagging"
    ar_ref=$image:$tag
else
    ar_ref=asia-southeast1-docker.pkg.dev/til-ai-2025/$TEAM_NAME-repo-til-25/$image:$tag
    echo "Tagging '$image:$tag' as '$ar_ref'..."
    docker tag $image:$tag $ar_ref
fi

echo "Pushing '$ar_ref' to Artifact Registry..."
docker push $ar_ref && \
echo "Submitting '$ar_ref' for automatic evaluation..." && \
gcloud ai models upload --region asia-southeast1 --display-name "$TEAM_NAME-$task" \
    --container-image-uri $ar_ref --container-health-route /health --container-predict-route /$task \
    --container-ports $port --version-aliases default
