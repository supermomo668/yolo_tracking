curl -X 'POST' \
  'http://127.0.0.1:8000/track' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@assets/images/dog_cat.jpg'