nohup celery -A app.celery worker -P gevent -c 1 -Q dataset,generation,mail --loglevel INFO > Worker.log 2>&1 &
