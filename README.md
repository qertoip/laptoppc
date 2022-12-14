Laptop vs PC image classifier using Keras
=========================================

This is example use of:

* TensorFlow / Keras
* Convolutional Neural Networks
* Transfer Learning (building on top of existing model, EfficientNetV2B0)

...to reliably tell apart laptops from PC-s in a real photographs.

Demo
----

See it in action at: https://ml.qertoip.com/

Installation
------------

After cloning the repository, please run:

    cd laptoppc
    poetry install

Running
-------

Trained model is embedded in the project so you can go straight to running a webapp if you like. 

To run webserver locally:

    poetry shell
    python laptoppc/webserver.py    # .venv is created by `poetry install` somewhere

Then go to: http://127.0.0.1:5000/

Modifying
---------

Trained model is embedded in the project so you can go straight to running these commands if you like.

To print model summary, run `python laptoppc/print_model.py`

To train model from data, run `python laptoppc/train_model.py`

To recreate data directory, uncomment and run `python laptoppc/create_dataset.py` (not recommended because you will need to manually cleanup the dataset).
