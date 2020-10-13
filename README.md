# frameSharpening

## create a virtual environment
*make sure you are in the frameSharpening directory*

## For Python 2 use
    python -m venv env

## For Python 3 use
    python3 -m venv env  

## activate it
windows: 

    env/Scripts/activate

linux: 

    source env/bin/activate

## update pip
windows: 
    
    python -m pip install -U pip

linux: 
    
    pip install -U pip

## install requirements
    pip install -r requirements.txt

## run flask app
    flask run