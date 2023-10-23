## Serve Usage

You may use the following endpoint to enable tracking for an object.

Test if the endpoint is live:
### Test 
```
curl -X 'GET' \
  'https://pleasant-large-kingfish.ngrok-free.app/' \
  -H 'accept: application/json'
```

### Usage

<b> An online copy of the API documentation can be found [here](https://pleasant-large-kingfish.ngrok-free.app/doc)</b> but detailed examples are given below, which is also available [here](https://pleasant-large-kingfish.ngrok-free.app/docv2)

To refresh the tracker for a different scene:
* refresh tracker
```
curl -X 'GET' \
  "https://pleasant-large-kingfish.ngrok-free.app/new' \
  -H 'accept: application/json'
```
To track the image continuously, call the following endpoint with your frames. 
Note the format of the post request for the timestamp must follow `'%Y-%m-%d %H:%M:%S'`
* track image
```
curl -X 'POST' \
  "https://pleasant-large-kingfish.ngrok-free.app/track" \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@assets/images/dog_cat.jpg' \
  -F 'timestamp="2023-10-23 14:45:30"
```
The result of track gives the following information:
{"result":
  id<int>: Increment of the current track
  timestamp<str>: timestamp
  track_id<list>: the id of the tracked object
  duration<list>: time spent by the object in the format of '%d%H:%M:%S'
  box<list>: box in `xywh`
  centroids<list>: centroids of the object
  labels<list>: labels of the box
  confidence<list>: confidence
}
```
{
  "results": [
    {
      "id": 1,
      "timestamp": "2023-10-21 06:11:50",
      "track_id": [
        1,
        2,
        3,
        4,
        5
      ],
      "duration": [
        0:00:00:01,
        0:00:00:01,
        0:00:00:01,
        0:00:00:01,
        0:00:00:01,
      ],
      "box": [
        [
          895.691650390625,
          808.6295776367188,
          321.033203125,
          536.0555419921875
        ],
        [
          82.24114990234375,
          955.2718505859375,
          164.1314697265625,
          238.367431640625
        ],
        [
          563.54541015625,
          241.19091796875,
          190.4150390625,
          328.730224609375
        ],
        [
          1530.6856689453125,
          964.37939453125,
          289.153564453125,
          197.07421875
        ],
        [
          526.0513916015625,
          975.7635498046875,
          229.6409912109375,
          206.508544921875
        ]
      ],
      "centroids": [
        [
          895.691650390625,
          808.6295776367188
        ],
        [
          82.24114990234375,
          955.2718505859375
        ],
        [
          563.54541015625,
          241.19091796875
        ],
        [
          1530.6856689453125,
          964.37939453125
        ],
        [
          526.0513916015625,
          975.7635498046875
        ]
      ],
      "labels": [
        0,
        1,
        0,
        0,
        0
      ],
      "confidence": [
        0.8641559481620789,
        0.8378390669822693,
        0.8312875032424927,
        0.8232210278511047,
        0.7460027933120728
      ]
    }
  ]
}
```

# Database Insertion

The above tracking information is added and committed into the database automatically at start with the given credentials for forming the SQL connection string, namely:
```
MYSQL_DATABASE
MYSQL_USER
MYSQL_PASSWORD
MYSQL_HOST
MYSQL_PORT
```
accordingly and should be self-explanatory. It assumes that the pk for the database is auto-incremental , otherwise it should encounter duplicate object error.

