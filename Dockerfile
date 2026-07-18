FROM python:3.11
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /code/requirements.txt
RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
