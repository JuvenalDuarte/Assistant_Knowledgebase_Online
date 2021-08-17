FROM python:3.8

RUN mkdir /app
RUN mkdir /app/cfg
WORKDIR /app
ADD requirements.txt /app/
RUN pip install -r requirements.txt

ADD . /app

# Download the list of stopwords from github repository
RUN wget https://raw.githubusercontent.com/JuvenalDuarte/portuguese_stopwords/main/stopwords.txt --no-verbose -P /app/cfg

RUN rm -rf tmp

EXPOSE 5000

CMD gunicorn -c /app/gunicorn.conf.py main:application
