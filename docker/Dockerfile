# Build TANDEM-DIMPLE container
# This container is built over the mathworks/matlab:r2023a image
# This container is used to run the TANDEM-DIMPLE software

# Build the TANDEM-DIMPLE container
# docker build -t tandem -f docker/Dockerfile .

# Use the official MATLAB image as a parent image
FROM mathworks/matlab:r2023a

USER root

# Install required dependencies, git
RUN apt update --quiet \
    && apt install --yes --quiet software-properties-common \
    && apt install --yes --quiet git wget gcc g++ make \
    && apt install --yes --quiet csh libboost-all-dev \
    && apt install --yes --quiet git 

# Install python 3.9
# Dependencies for python 3.9: openmm=8.1.0 prody propka3 matlab engine numpy pandas tensorflow biopython requests
RUN apt install --yes --quiet python3.9 python3-pip python3.9-venv python3.9-dev

# Create a virtual environment for the TANDEM-DIMPLE software
RUN python3.9 -m venv /tandem_venv 
ENV PATH="/tandem_venv/bin:$PATH"

# Update pip to the latest version
RUN pip3 install --upgrade pip

# Copy the TANDEM-DIMPLE source code to the container
# And set the working directory to the source code
COPY . /app/tandem
WORKDIR /app/tandem

# Install the Python dependencies TANDEM-DIMPLE 
RUN pip3 install -r requirements.txt
RUN pip3 install --no-cache-dir -e /app/tandem/rhapsody
RUN pip3 install --no-cache-dir -e /app/tandem/pdbfixer-master

# Test the installation : run /app/tandem/main.py
CMD ["python3", "/app/tandem/main.py"]
