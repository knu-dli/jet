# reproduce-paper

Makes the tables and figures of the paper from the finished runs, one script each. The scripts only read the outputs
of the other folders (no GPU or API call). The run id of each row is set at the top of each script. Our run ids use
older names: `sis` for JET, `plain` for Truncated SFT, `own` for Self, `bats` for Budget Tracker, and `sufficiency`
for Autorater.

```bash
cd reproduce-paper
bash run.sh                      # every table and figure -> out/
```

## File structure

```
reproduce-paper/
├── figure2.py                   # forced-answer score and trajectories still running at each step
├── table2.py                    # main results: Score, Calls, Cost, and the LaTeX of the table
├── table3.py                    # golden step ablation (n/2, 2n/3)
├── figure4.py                   # questions that reach the step limit
├── figure5.py                   # tool calls and score for each halting weight λ
├── table4.py                    # hyperparameters, read from the code and the saved run arguments
├── table5.py                    # mean score of each judge criterion
├── checkpoints.py               # validation score of every checkpoint, with the chosen one marked
├── figure1.txt                  # source files of the example in Figure 1
└── run.sh
```
