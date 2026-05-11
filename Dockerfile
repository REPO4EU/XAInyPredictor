
# Base image
FROM python:3.12-slim AS builder

RUN mkdir /src
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
COPY . /src/

## Execute any command that is needed in the Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

#Install the requirements and the module
RUN . /opt/venv/bin/activate && pip install --no-cache-dir --upgrade pip setuptools && pip install --no-cache-dir /src/

# Arguments needed for the labelling, provided from the docker build command (scripts/build.sh)
ARG BUILD_DATE
ARG BUILD_VERSION
ARG VCS_REF
ARG CI_PROJECT_NAME
ARG MAIN_TAG

# LABELS as per http://label-schema.org/rc1/
LABEL org.label-schema.org.label-schema.schema-version="1.0"
LABEL org.label-schema.version=$BUILD_VERSION
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.name=$CI_PROJECT_NAME

## Change following labels as pertinent, this is only informative.
LABEL org.label-schema.maintainer="Joaquim Aguirre Plans"
LABEL org.label-schema.email="quim.aguirre@hotmail.com"
LABEL org.label-schema.description="ShinyRAI Dockerfile"
# "${MAIN_TAG}"

## docker run will automatically run the tool using the entrypoint main
## it can be useful when building tools
## DON'T use ENTRYPOINT since it can cause issues when running tools in nextflow
CMD ["echo", "Wrong Docker Image built."]

# Actual Image to be used
FROM python:3.12-slim

# copy only Python packages to limit the image size
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["/opt/venv/bin/shiny", "run", "/opt/venv/lib/python3.12/site-packages/ShinyRAI/app.py", "--host", "0.0.0.0", "--port", "8000"]

