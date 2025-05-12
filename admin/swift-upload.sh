#!/bin/bash
#1) Download open-rc from nectar dashboard
#2) Set password in nectar dashboard -> Settings
source access-nri-store-openrc.sh
pip install python-swiftclient python-keystoneclient
#
#swift list

#Fixing flipped jpegs
#sudo apt install libjpeg-turbo-progs
#mkdir tmp
#mv *.jpg tmp
#cd tmp
#for i in *.jpg; do jpegtran -flip vertical $i > ../$i; done
pushd /media/data/accessvis
URL=https://object-store.rc.nectar.org.au/v1/
swift --os-storage-url ${URL}AUTH_${OS_PROJECT_ID} upload accessvis bluemarble/cubemap_2048
swift --os-storage-url ${URL}AUTH_${OS_PROJECT_ID} upload accessvis bluemarble/cubemap_4096
swift --os-storage-url ${URL}AUTH_${OS_PROJECT_ID} upload accessvis bluemarble/cubemap_8192
swift --os-storage-url ${URL}AUTH_${OS_PROJECT_ID} upload accessvis bluemarble/cubemap_16384
#swift --os-storage-url ${URL}AUTH_${OS_PROJECT_ID} upload accessvis bluemarble/source_full
swift --os-storage-url ${URL}AUTH_${OS_PROJECT_ID} upload accessvis relief
swift --os-storage-url ${URL}AUTH_${OS_PROJECT_ID} upload accessvis landmask
swift --os-storage-url ${URL}AUTH_${OS_PROJECT_ID} upload accessvis gebco/*.npz
popd
