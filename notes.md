# NOTES

questiosn we need to answer:

1. where specifically are the check-points stored?
    - checkpoints are not used. resume functionality based on what we have in the DB vs what is in the kaggle json file. note this is a solution that is specific to this workflow.
2. where specifically are the yaml files for each configurable part of this workflow
    - we will need to make our core.workflows.workflow_arxiv_parallel method more production ready by moving all of its configurables to a yaml file.
3.

---

## command to launch the data integration of arxiv data into the graph

```python
python -m core.workflows.workflow_arxiv_parallel --count 1000 --batch-size 100 --embedding-batch-size 64 --workers 2 --drop-collections
```

---

## After rebuilding the complete graph with all 2.82M papers, you'll need to

  1. Extract features for the complete graph (the current features.npz only has 1.68M papers)
  2. Relaunch training with the complete dataset

  Here's the sequence:

### Step 1: Rebuild complete graph (with monitoring)

  cd /home/todd/olympus/HADES-Lab/tools/graphsage
  python rebuild_complete_graph.py

### Step 2: Extract features for ALL papers

  python extract_features.py

### Step 3: Relaunch distributed training (with monitoring)

### Terminal 1 - Training

  python train_distributed.py

### Terminal 2 - Monitoring

  python monitor_graphsage.py --mode training

  The training will now:

- Use all 2.82M papers (not just 1.68M)
- Have proper edge connections for every paper
- Capture complete metrics for your paper

  Your previous training reached 85% accuracy on the incomplete graph at epoch 9. With the complete
  graph, you should see:

- Better coverage across all academic disciplines
- More meaningful embeddings (no orphaned papers)
- Complete computational cost metrics for the "librarian gatekeeper" creation

  The monitoring will capture the full "translation" process in ANT terms - transforming the complete
  2.82M paper graph into navigable embedding space.

---
