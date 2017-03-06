FROM heroku/miniconda

# Grab requirements.txt.
ADD ./script/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

# Add our code
ADD ./script /opt/script/
WORKDIR /opt/script

RUN conda install scikit-learn
