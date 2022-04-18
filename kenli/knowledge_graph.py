import json
import random
import subprocess
from typing import List, Dict


def exec_bash(s):
    subprocess.run(s, shell=True)


class KnowledgeGraph:
    def get_tokens(self, premise: str, hypothesis: str) -> List[str]:
        raise NotImplementedError('get_tokens is not implemented!')


class CauseNet(KnowledgeGraph):
    max_depth: int
    min_object_count: int
    g: Dict[str, List[str]]

    def __init__(self, max_depth: int = 2, min_object_count: int = 5, use_full: bool = False):
        self.max_depth = max_depth
        self.min_object_count = min_object_count
        
        print('Downloading CauseNet...')
        if use_full:
            exec_bash('wget https://groups.uni-paderborn.de/wdqa/causenet/causality-graphs/causenet-full.jsonl.bz2')
            exec_bash('mv causenet-full.jsonl.bz2 causenet.jsonl.bz2')
        else:
            exec_bash('wget https://groups.uni-paderborn.de/wdqa/causenet/causality-graphs/causenet-precision.jsonl.bz2')
            exec_bash('mv causenet-precision.jsonl.bz2 causenet.jsonl.bz2')
        print('Resulting size of causenet.jsonl.bz2:')
        exec_bash('du -h causenet.jsonl.bz2')

        print('Removing old file...')
        #exec_bash('rm causenet.jsonl')
        print('Unpacking bz2 file...')
        exec_bash('bzip2 -d causenet.jsonl.bz2')
        print('Resulting size of causenet.jsonl:')
        exec_bash('du -h causenet.jsonl')

        print('Extracting connections...')
        connections = []
        with open('causenet.jsonl') as f:
            for line in f.readlines():
                connections.append(json.loads(line)['causal_relation'])
        print('Amount of edges:', len(connections))

        print('Building graph...')
        self.g = {}
        for edge in connections:
            self.g[edge['cause']['concept']] = self.g.get(edge['cause']['concept'], []) + [edge['effect']['concept']]
        print('Amount of vertexes:', len(self.g))
    
    def get_tokens(self, premise: str, hypothesis: str) -> List[str]:
        premise_objects = self._filter_objects(self._get_objects(premise))
        hypothesis_objects = self._filter_objects(self._get_objects(hypothesis))
        useful_objects, edges = self._get_useful_objects(premise_objects, hypothesis_objects)
        #print(f'Added {len(useful_objects)} useful objects')
        if len(premise_objects) == 0:
          useful_objects.update({"cat"})
        else:
          rand_steps = - len(useful_objects) + self.min_object_count
          for _ in range(rand_steps):
              rand_objects, rand_edges = self._get_random_path(premise_objects)
              useful_objects.update(rand_objects)
              edges.update(rand_edges)
        return useful_objects, edges

    def _get_objects(self, sentence: str):
        sentence = sentence.lower()
        words = []
        cur_word = ""
        for c in sentence:
            if c.isalpha():
                cur_word += c
            else:
                if cur_word != "":
                    words.append(cur_word)
                    cur_word = ""
        if cur_word != "":
            words.append(cur_word)
        result = words
        for i in range(1, len(words)):
            result.append(words[i - 1] + '_' + words[i])
        return result
    
    def _filter_objects(self, objects):
        return set(filter(lambda obj: obj in self.g, objects))
    
    def _get_useful_objects(self, premise_objects, hypothesis_objects):
        result = set()
        edges = set()

        cur_row = []
        def dfs(obj, cur_d):
            cur_row.append(obj)
            if obj in hypothesis_objects:
                #print('Found path!', cur_row)
                result.update(cur_row)
                for edge_ind in range(len(cur_row) - 1):
                  edges.add((cur_row[edge_ind], cur_row[edge_ind + 1]))
            for nxt_obj in self.g.get(obj, []):
                if cur_d + 1 <= self.max_depth:
                    dfs(nxt_obj, cur_d + 1)
            cur_row.pop()
        
        for premise_obj in premise_objects:
            if premise_obj not in hypothesis_objects:
                dfs(premise_obj, 0)

        return result, edges
    
    def _get_random_path(self, premise_objects):
        edges = set()
        result = [random.choice(list(premise_objects))]
        while len(result) < self.max_depth:
            result.append(random.choice(self.g[result[-1]]))
        for edge_ind in range(len(result) - 1):
            edges.add((result[edge_ind], result[edge_ind + 1]))
        #print('Added random path:', result)
        return result, edges

