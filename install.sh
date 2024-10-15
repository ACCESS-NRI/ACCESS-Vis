#!/bin/bash
if [[ $HOSTNAME == *"gadi.nci.org.au"* ]]; then
  echo "Setting up for user on gadi..."
  cd /g/data/nf33/public/data/accessvis/
  module use /g/data/hh5/public/modules
  module load conda/analysis3

  #Install kernels for GPU and CPU rendering

  source vis_venv/bin/activate
  python3 -m ipykernel install --user --name vis-venv --display-name "Visualisation + conda/analysis3"
  deactivate

  #For CPU visenv
  source vis_venv_nogpu/bin/activate
  python3 -m ipykernel install --user --name vis-venv-nogpu --display-name "Visualisation no GPU + conda/analysis3"
  deactivate

else
  echo "Installing locally..."
  python -m pip install --editable .
fi


