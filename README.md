# Calculating Directionality of Variables using Masked Language Models
This application finds the probability of each effect token to appear at the end of a sentence which has a causal token with verb.

For example if **input_tokens=[education, income]**, this application will
find probability of the token `income to occur at the end of :
        education improves ______ etc.
        Note that if either of the tokens are multiworded tokens (e.g.; rice production) probabilities are recursively
        averaged. For example, for input tokens=["income", "rice production"] overall probability of rice production to
        occur at the end of income based sentences are:average(prob(income promotes rice [MASK]), prob(income promotes [MASK])

## To execute this application
 ```
    conda create -n directionality python=3
    conda activate directionality
    pip install -r requiements.txt    
    python direction_validation.py    
```
## Inputs

There are 3 types of inputs expected to this code. 

- causal variables
- effect variables
- verbs


Causal and effect variables, should be in the file 
`data/query_directionality_variables.csv`

Causal variables should be separated by comma, which inturn should be separated from effect variables using a pipe `|`

e.g.,

`stability|income`

`stability,education level,level of education|income,level of income`

The verbs should be provided in the file data/verbs.py

Note: In this version (v1.0) even though we provide verbs which are not
`promotes` or `inhibits` like, (e.g.,all_does_not_cauase_verbs) they are 
not used in probability calculations.

## Outputs

The outputs are the probabilities of each combination of the aforementioned
variables to occur with the list of provided verbs

The outputs will be found in `outputs/prob.json`

### Logging

Log files which can be used for debugging purposes and sentence level
 probability details can be found in `logs/*.log.`

Names of log files will include the input variables, type of masked language model used,
and the date of execution of this program.

e.g.,`numberofyearsoffarming_income_distilbert-base-uncasedMar-15-2021.log`

Logging levels can be changed using the variable LOGGING_LEVEL in `./direction_validation.py`


