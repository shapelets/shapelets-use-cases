

# Spain energy price prediction

This project is a use case to predict the price of energy using the spanish energy data. 


# Requisites

If you are using the Shapelets docker image, you need to install `LightGBM` and `XGBoost`.

You can do it entering the docker image and running pip, like this

```
your_machine $ docker ps

CONTAINER ID   IMAGE                                         COMMAND                  CREATED          STATUS                    PORTS                                         NAMES
498cc3ea7a59   shapeletsdev/shapelets-solo:0.5.2.dev220607   "/bin/sh -c 'python3…"   1 minutes ago   Up 1 minutes             0.0.0.0:443->443/tcp, :::443->443/tcp         gracious_mayer


your_machine $ docker exec -it 498cc3ea7a59 /bin/bash

docker_container $ pip install lightgbm xgboost
```


# Launch


Launch this dataapp its simple, like all in Shapelets! Just execute the python script.

```
docker_container $ cd io/_your_folder_/shapelets-use-cases/spain-energy-price-prediction
docker_container $ python main.py
```

