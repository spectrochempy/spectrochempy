#!/usr/bin/env bash

echo "Installing SpectroChemPy ... "
rm -rf spectrochempy &> output.log
git clone https://github.com/spectrochempy/spectrochempy.git &>> output.log
pip install -U -r ./spectrochempy/requirements.txt &>> output.log
cd ./spectrochempy || exit
python setup.py install &>> output.log
cd ..

echo "Installing testdata ... "
VERSION="1.1" &>> output.log
SCP_DATA_GZ="$VERSION.tar.gz" &>> output.log
(wget -c "https://github.com/spectrochempy/spectrochempy_data/archive/$SCP_DATA_GZ" -O - | tar -xz) &>> output.log

SCP_DATA="spectrochempy_data-$VERSION" &>> output.log
mkdir -p /root/.spectrochempy &>> output.log
rm -rf /root/.spectrochempy/testdata &>> output.log
mv $SCP_DATA/testdata  /root/.spectrochempy &>> output.log
rm -rf $SCP_DATA &>> output.log

rm -f set_colab.sh &>> output.log
rm -rf sample_data &>> output.log

echo "SpectroChemPy is now installed."
echo "(But in case of problems, you may have a look to the logs; type: cat output.log)"
