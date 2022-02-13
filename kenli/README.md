## Example of CauseNet usage
```python
cause_net = CauseNet()
cause_net.get_tokens('vegan diet leads to poor nutrition', 'vegan diet leads to diabetes')
```

Result:

```
{'depression', 'anxiety', 'inflammation', 'insomnia', 'diabetes', 'health_problems', 'injury', 'problems', 'changes', 'childhood_obesity', 'conditions', 'disease', 'poor_health', 'overweight', 'type_2_diabetes', 'nutrition', 'weight_gain', 'stress', 'imbalances', 'poor_nutrition', 'heart_disease', 'increase', 'cancer', 'diseases', 'illnesses', 'illness', 'obesity', 'cardiovascular_disease', 'condition', 'symptoms'}
```