#!/usr/bin/env python
# Это скриптовая версия оценки качества модели. Версия колаба доступна на:
#  https://colab.research.google.com/drive/1QuWxJYNnUaCmZSOWFdZnr9mghk6e7hLy
# Отличий много, к примеру здесь не указаны примеры строк в датасетах


import argparse
import json
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameters for training and modes for evaluating model')
    parser.add_argument('--batch_size',    type=int,   default=10,     help='batch size')
    parser.add_argument('--lr',            type=float, default=2e-5,   help='learning rate')
    parser.add_argument('--workers',       type=int,   default=1,      help='num workers for dataloader')
    parser.add_argument('--lamb',          type=float, default=1.0,    help='regularizer lambda')
    parser.add_argument('--precision',     type=int,   default=16,     help='precision for pytorch_lightning.Trainer')
    parser.add_argument('--gpus',          type=str,   default='0,',   help='gpus for pytorch_lightning.Trainer')
    parser.add_argument('--max_epoch',     type=int,   default=1,      help='max_epoch for pytorch_lightning.Trainer')

    parser.add_argument('--tasks',         type=str,   default='snli,sick,anli,mnli', help='nlp datasets to evaluate, separated by commas (default: all)')
    parser.add_argument('--roberta_model', type=str,   default='base', help='model of roberta to use (base or large)')
    
    parser.add_argument('--prepare',    dest='prepare', action='store_true',  help='should prepare model evaluation')
    parser.add_argument('--no-prepare', dest='prepare', action='store_false', help='should not prepare model evaluation')
    parser.set_defaults(feature=True)
    
    args = parser.parse_args()
    
    # Проверка, что tasks и roberta_model корректны
    allowed_tasks = {'snli', 'sick', 'anli', 'mnli'}
    for task in args.tasks.split(','):
        if task not in allowed_tasks:
            raise ValueError(f'Found unknown task: {task}')
            
    allowed_roberta_model = ['base', 'large']
    if args.roberta_model not in allowed_roberta_model:
        raise ValueError(f'Found unknown roberta model: {args.roberta_model}')
    return args
    

def exec_bash(s):
    subprocess.run(s, shell=True)

    
# Скачиваем и устанавливаем всё необходимое
def prepare(args):
    # Удаляем ненужный sample_data
    exec_bash('rm -rf sample_data')
    
    # Клонируем наш репозиторий
    exec_bash('git clone https://github.com/HSE-projects/logical_reasoning')
    
    # Удаляем torchtext так как в модели хотят версию торча 1.6, а она не поддерживается с новым torchtext, torchvision, ...
    exec_bash('pip uninstall torchtext -y')
    
    # Устанавливаем все необходимые пакеты
    exec_bash('pip install -r ./logical_reasoning/eval_model/Self_Explaining_Structures_Improve_NLP_Models/requirements.txt')
    exec_bash('pip install datasets')
    exec_bash('pip install gdown')
    
    exec_bash('apt-get install git-lfs')
    exec_bash('git lfs install')
    
    # Подгружаем модель и её чекпоинт
    if args.roberta_model == 'base':
        exec_bash('git clone https://huggingface.co/roberta-base')
        exec_bash('gdown https://drive.google.com/uc?id=1bJVYekaOVJjI2woxqaqKMhWP0HLau105')  # SNLI Base
        exec_bash('mv ./epoch=4-valid_loss=-0.6472-valid_acc_end=0.9173.ckpt ./snli_baseline.ckpt')
    else:
        exec_bash('git clone https://huggingface.co/roberta-large')
        exec_bash('gdown https://drive.google.com/uc?id=1AHMGgYFqcr3NmA-py6AMsO83w8FeNSJn')  # SNLI Large
        exec_bash('mv ./epoch=2-valid_loss=-0.2620-valid_acc_end=0.9223.ckpt ./snli_baseline.ckpt')


# Меняем конфиг, чтобы был `num_labels` = 3
def change_config(args):
    config_path = f'./roberta-{args.roberta_model}/config.json'

    config = json.load(open(config_path))
    config['num_labels'] = 3
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


class cd:
    '''Context manager for changing the current working directory'''
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


# Код для дообучения модели для произвольной задачи
def train_for_task(args, task):
    bert_path = os.getcwd() + f'/roberta-{args.roberta_model}'
    ckpt_path = os.getcwd() + f'/{task}_ckpt'
    
    with cd('./logical_reasoning/eval_model/Self_Explaining_Structures_Improve_NLP_Models/explain'):
        subprocess.run([
            'python', 'trainer.py',
            '--bert_path', bert_path,
            '--task', task,
            '--save_path', ckpt_path,
            '--gpus', args.gpus,
            '--mode', 'train',
            '--precision', str(args.precision),
            '--lr', str(args.lr),
            '--batch_size', str(args.batch_size),
            '--lamb', str(args.lamb),
            '--workers', str(args.workers),
            '--max_epoch', str(args.max_epoch)
        ])


# Код для оценивания модели для произвольной задачи
def eval_task(args, task):
    bert_path = os.getcwd() + f'/roberta-{args.roberta_model}'
    ckpt_path = os.getcwd() + (f'/{task}_ckpt/last.ckpt' if task != 'snli' else '/snli_baseline.ckpt')
    save_path = os.getcwd() + f'/{task}_result'
    data_path = os.getcwd() + '/snli_1.0'  # SNLI only
    
    exec_bash(f'mkdir {save_path}')

    with cd('./logical_reasoning/eval_model/Self_Explaining_Structures_Improve_NLP_Models/explain'):
        subprocess.run([
            'python', 'trainer.py',
            '--bert_path', bert_path,
            '--task', task,
            '--checkpoint_path', ckpt_path,
            '--save_path', save_path,
            '--gpus', args.gpus,
            '--mode', 'eval',
            '--data_dir', data_path  # SNLI only
        ])


if __name__ == '__main__':
    print('Parsing args...')
    args = parse_args()
    print('Parsed args')
    
    if args.prepare:
        print('Preparing for evaluation')
        prepare(args)
        print('Ready to evaluate')
    change_config(args)
    
    for task in args.tasks.split(','):
        if task == 'snli':
            if not os.path.exists('./snli_1.0'):
                print('Downloading SNLI dataset')
                exec_bash('wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip')
                exec_bash('unzip snli_1.0.zip')
                exec_bash('rm -rf __MACOSX')
                exec_bash('rm snli_1.0.zip')
                print('Downloaded SNLI dataset')
        else:
            print('Fine-tuning checkpoint to evaluate task', task)
            train_for_task(args, task)
            print('Done fine-tuning')
        print('Evaluating task', task)
        eval_task(args, task)
        print('Done evaluating task', task)
    print('Done')
