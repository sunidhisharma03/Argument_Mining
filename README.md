# Legal Argument Mining Classifier

A simple BERT-based classifier to identify legal argument components from ECHR case texts.

## Labels

- **0**: Non-Argument  
- **1**: Premise  
- **2**: Conclusion

## Model

Fine-tuned on the [ECHR corpus](http://www.di.uevora.pt/~pq/echr/) using `nlpaueb/legal-bert-base-uncased`.

## How to Run

```bash
pip install -r requirements.txt
python argument_classifier_demo.py
