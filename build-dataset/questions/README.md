# questions

The RL questions of SWE-QA-Pro, which the Self setting uses for rollouts. Each question has a repository, a commit, and
a ground truth answer. Of the 464 questions, 400 are for training and 60 for validation (`out/split.json`). The split
keeps the questions of one repository on one side, and leaves out the questions whose repository no longer exists or
also appears in a test benchmark.

```bash
cd build-dataset
python -m questions.clone      # clones the repositories at their commits into out/repos-rl/ (about 40 GB)
python -m questions.split      # only to rebuild out/split.json; the included file is the split we used
```

## Files

```
questions/
├── load.py        # reads the RL questions
├── clone.py       # clones their repositories
├── split.py       # training and validation split
└── out/split.json
```
