curl -X POST -H 'Content-Type: application/json' localhost:5000/predict -d \
'
	[	
		{"sepal_length":1, "sepal_width":1, "petal_length":1, "petal_width":1},
		{"sepal_length":4, "sepal_width":4, "petal_length":4, "petal_width":5}
	]
'
curl -X POST -H 'Content-Type: application/json' localhost:5000/predict -d \
'
        [
                {"badname":1, "sepal_width":1, "petal_length":1, "petal_width":1},
                {"badname":4, "sepal_width":4, "petal_length":4, "petal_width":5}
        ]
'
