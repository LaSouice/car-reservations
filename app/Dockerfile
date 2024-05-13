# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000
EXPOSE 5000

# Define environment variable
ENV MODEL_FILE data/06_models/regressor.pickle

# Run app.py when the container launches
CMD ["python", "app.py"]