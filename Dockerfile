FROM aimle2e/nvidia-pytorch:23.06-py3-03

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install production dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt


# Copy local code to container image
COPY codellama-container.py ./

CMD ["python", "codellama-container.py"]