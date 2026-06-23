# create virtual env using venv
python -m venv env

# activate
source ./env/bin/activate

# install reqs
sudo apt-get update
sudo apt-get install -y python3-opencv
pip install -r requirements.txt

# deactivate
deactivate 