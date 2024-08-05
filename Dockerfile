FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Install additional packages
RUN apt-get -y update && \
         apt-get -y upgrade && \
         apt-get install -y python3-pip python3-dev python3-venv 

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install the necessary packages
RUN pip3 install tensorflow==2.10.0 opencv-python-headless matplotlib scikit-learn

# Create segment patches
RUN mkdir -p gq-segmented-terrain-classes
RUN mkdir -p gq-segmented-terrain
RUN mkdir -p gq-segmented-road-classes
RUN mkdir -p gq-segmented-road
RUN mkdir -p OUTPUT

# Sets correct directory
WORKDIR /SemanticSeg

# Copy application code to working directory
COPY TrainImgsOne /SemanticSeg/TrainImgsOne
COPY MaskPatchesOne /SemanticSeg/MaskPatchesOne
COPY alc-resized /SemanticSeg/alc-resized
COPY unet_model.py .
COPY image_processing.py .
COPY generator_main.py .
COPY model_weights.py .
COPY test.py .
COPY segmented_concat.py .
COPY merge_imgs.py .
COPY run.sh .

# Runs the run.sh script
ENTRYPOINT [ "./run.sh" ]

