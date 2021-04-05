# Calculating Directionality of Variables using Masked Language Models
This application finds the probability of each effect token to appear at the end of a sentence which has a causal token with verb.

For example if **input_tokens=[education, income]**, this application will
find probability of the token `income to occur at the end of :
        education improves ______ etc.
        Note that if either of the tokens are multiworded tokens (e.g.; rice production) probabilities are recursively
        averaged. For example, for input tokens=["income", "rice production"] overall probability of rice production to
        occur at the end of income based sentences are:average(prob(income promotes rice [MASK]), prob(income promotes [MASK])

## Pre requisites
 ```
    conda create -n directionality python=3
    conda activate directionality
    pip install -r requirements.txt   
    mkdir outputs 
    python  python calc_relation_probabilities.py "data/inputs.tsv"    
```

## To execute this application

`python  python calc_relation_probabilities.py [input_file]`

For example:
 `python  python calc_relation_probabilities.py "data/inputs.tsv"    `
## Inputs
######todo: update based on final data format you decide on

There are the inputs expected to this code. 

- causal variables
- effect variables
- verbs
- masked language model


Causal and effect variables, should be in the file 
`data/inputs.tsv`


Causal variables should be separated by comma, which inturn should be separated from effect variables using a pipe `|`

e.g.,

`stability|income`

`stability,education level,level of education|income,level of income`

The verbs should be provided in the file data/verbs.py

Note: In this version (v1.0) even though we provide verbs which are not
`promotes` or `inhibits` like, (e.g.,all_does_not_cauase_verbs) they are 
not used in probability calculations.

By default distilbert is used as the masked language model. You can change this by changing the value of LIST_MODEL_NAME in `./direction_validation.py`

## Outputs
######todo: update based on final data format you decide on

The outputs are the probabilities of each combination of the aforementioned
variables to occur with the list of provided verbs

The outputs will be found in `outputs/prob.json`

### Logging

Log files which can be used for debugging purposes and sentence level
 probability details can be found in `logs/*.log.`

Names of log files will include the input variables, type of masked language model used,
and the date of execution of this program.

e.g.,`numberofyearsoffarming_income_distilbert-base-uncasedMar-15-2021.log`

Logging levels (e.g., DEBUG, INFO, WARNING, ERROR etc.) can be set using the variable LOGGING_LEVEL in `./direction_validation.py`


