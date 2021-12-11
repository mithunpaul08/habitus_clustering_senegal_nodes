# Grounding descriptions of graphs using trained RTE models

## Pre requisites
 ```
    conda create -n habitus python=3
    conda activate habitus
    pip install -r requirements.txt   
    mkdir outputs 
    mkdir logs
    cd data
    wget https://osf.io/8rcxu/download .
    cd ..    
        
```

## To execute this application

`python get_pathways_rte.py`

### Logging

Log files which can be used for debugging purposes and sentence level
 probability details can be found in `logs/*.log.`




