# sturdy-couscous
Simplifying Amplitudes using ML

We recommend using a virtual environment, whose setup is described in the `environment` folder. 
Ensure this venv is activated, then follow the instructions below to generate data, then train a Transformer on the data.

### Data Generation Instructions
...

### Transformer Training Instructions
In the main repo directory run:
```
python3 transformer/transformer_trainer.py
```
...to train a transformer model with the hyperparameters specified in that script. To then evaluate the model, run:
```
python3 transformer/transformer_evaluator.py
```
...which has its own hyperparameters, including a choice between greedy decoding, beam search, and nucleus sampling regimes.

## BibTeX Citation  
``` 
raise NotImplementedError("Paper yet to be published.")
```
