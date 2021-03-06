# Результаты find_ideal_query

Результаты хранятся в папках `{dataset}_{seed}`. К примеру `mnli_42` означает, что использовался датасет MultiNLI и генерация примеров происходила при сиде 42

В каждой папке находится:
* `log_{gpu}.txt` - подробные логи запуска. К примеру можно посмотреть, сколько было итераций и на какой всё стабилизировалось
* `{dataset}_model_output_{seed}` - чекпоинт модели без файлов `pytorch_model.bin` и `optimizer.pt` (удалил, так как они много весят)
* `{dataset}_final_query` - итоговый датасет из хороших примеров
* `{dataset}_train_sample` - чекпоинты датасетов после добавления каждого примера
* `{dataset}_best_metrics.yaml` - итоговый `accuracy`

Датасеты читаются через `torch.load()`:
```python
import torch
torch.load('./snli_34/snli_final_query')
>>> Dataset({
    features: ['input_ids', 'attention_mask', 'labels'],
    num_rows: 52
})
```
