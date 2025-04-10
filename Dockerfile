FROM python:3.9.13

Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

Install production dependencies.
RUN pip install -r requirements.txt

EXPOSE 8080

CMD python main .py

