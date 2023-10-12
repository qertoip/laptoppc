Laptop vs PC image classifier using Keras
=========================================

This is example use of:

* TensorFlow / Keras
* Convolutional Neural Networks
* Transfer Learning (building on top of existing model, EfficientNetV2B0)

...to reliably tell apart laptops from PC-s in a real photographs.

Demo
----

See it in action at: https://ml.qertoip.com/laptoppc/

Installation
------------

Tested on Python 3.10.12.

After cloning the repository, please run:

    cd laptoppc
    bin/reset-venv  # creates .venv and installs dependencies

Running
-------

The trained model is embedded in the project,
so you can go straight to running a webapp. 

To run webserver locally:

    bin/dev-server

Then go to: http://127.0.0.1:5000/laptoppc/

Modifying
---------

The trained model is embedded in the project,
so you can go straight to running these commands.

Always start by activating venv: `source .venv/bin/activate`

To run some example predictions, run `python laptoppc/main_example.py`

To visualize NN, run `python laptoppc/main_visualize_model.py`

To train model from data, run `python laptoppc/main_train.py`

To recreate data directory, uncomment and run `python laptoppc/main_create_dataset.py` (not recommended because you will need to manually clean up the dataset).


Visualization of base vs finetuned
----------------------------------

See [model/base_model.png](model/base_model.png) vs [model/finetuned_model.png](model/finetuned_model.png). 
The final 3 layers differ between base and finetuned model, as expected.
